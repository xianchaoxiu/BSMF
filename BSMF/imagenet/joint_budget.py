"""Select ResNet-50 channels by DSF A-row importance under joint FLOPs and parameter budgets.

Channel counts refer to the dense model; costs follow THOP at 224x224 resolution."""
import math
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

BASE_FLOPS = 4133742592
BASE_PARAMS = 25557032


def blocks():
    in_channels = 64
    for stage, count in enumerate((3, 4, 6, 3), 1):
        width, out_channels, spatial = 64 * 2**(stage-1), 256 * 2**(stage-1), 56 // 2**(stage-1)
        for block in range(count):
            before_spatial = spatial * (2 if stage > 1 and block == 0 else 1)
            yield (f'layer{stage}.{block}', stage, block, in_channels, width, out_channels, before_spatial, spatial)
            in_channels = out_channels


def constant_cost():
    return (3*64*49*112**2 + 4*64*112**2 + 2048*50 + 2048*1000,
            3*64*49 + 2*64 + 2048*1000 + 1000)


def block_cost(info, a, b):
    name, stage, block, inc, width, out, ins, spatial = info
    params = inc*a + 9*a*b + b*out + 2*(a+b+out)
    flops = inc*a*ins**2 + (9*a*b + b*out)*spatial**2 + 4*(a*ins**2 + (b+out)*spatial**2)
    if block == 0:
        params += inc*out + 2*out
        flops += inc*out*spatial**2 + 4*out*spatial**2
    return int(flops), int(params)


def profile_cfg(cfg):
    if len(cfg) != 49 or cfg[0] != 64:
        raise ValueError('Expected a 49-value ResNet50 ImageNet cfg with protected stem.')
    flops, params = constant_cost()
    for i, info in enumerate(blocks()):
        inc, a, b = cfg[1+3*i:4+3*i]
        if inc != info[3] or not (1 < a <= info[4] and 1 < b <= info[4]):
            raise ValueError(f'Invalid original-channel subset for {info[0]}: {inc,a,b}')
        f, p = block_cost(info, a, b)
        flops += f; params += p
    return flops, params


def check_budget(flops, params, config):
    fd = 100 * (1 - flops / config['baseline_flops'])
    pd = 100 * (1 - params / config['baseline_params'])
    tf, tp, tol = config['flops_drop_percent'], config['params_drop_percent'], config['tolerance_pp']
    # One-sided: never exceed the requested absolute cost. Integer widths can
    # cause a slightly larger drop, bounded by 0.05 percentage points.
    if not (tf - 1e-9 <= fd <= tf + tol + 1e-9 and tp - 1e-9 <= pd <= tp + tol + 1e-9):
        raise RuntimeError(f'Joint budget miss: actual drops={fd:.6f}/{pd:.6f}, targets={tf}/{tp}, tol={tol}pp')
    return fd, pd


def select(scores, config, time_limit=45.0):
    infos = list(blocks())
    records = []
    # Optimize a per-block pair of keep counts. For each candidate keep count,
    # retain channels with the top-k DSF A-row scores.
    for block_index, info in enumerate(infos):
        name, stage, block, inc, width, out, ins, spatial = info
        sensitive = block == 0 or (stage == 4 and block == 1)
        medium = (stage == 1 and block in (1, 2)) or (stage == 2 and block == 1) or (stage == 3 and block in (1, 2))
        minimum = 0.50 if sensitive else 0.35 if medium else 0.15
        minimum_count = max(2, math.ceil(width * minimum))
        choices = np.unique(np.round(np.linspace(minimum_count, width, 21)).astype(int))
        cumul = []
        for conv in (1, 2):
            score = np.asarray(scores[f'{name}.conv{conv}'], dtype=np.float64)
            if score.shape != (width,) or not np.all(np.isfinite(score)) or np.any(score < 0):
                raise ValueError(f'Missing/invalid DSF A-row scores: {name}.conv{conv}')
            cumul.append(np.concatenate(([0.0], np.cumsum(np.sort(score)[::-1]))))
        for a in choices:
            for b in choices:
                f, p = block_cost(info, int(a), int(b))
                value = float(cumul[0][a] + cumul[1][b])
                records.append((block_index, int(a), int(b), f, p, value))
    n = len(records)
    row, col, data = [], [], []
    objectives = np.array([-r[5] for r in records])
    objectives /= max(1, abs(objectives).max())
    for j, (i, a, b, f, p, value) in enumerate(records):
        row.extend((i, 16, 17)); col.extend((j, j, j))
        data.extend((1.0, f/BASE_FLOPS, p/BASE_PARAMS))
    matrix = csc_matrix((data, (row, col)), shape=(18, n))
    const_f, const_p = constant_cost()
    retained_f = 1 - config['flops_drop_percent']/100
    retained_p = 1 - config['params_drop_percent']/100
    delta = config['tolerance_pp']/100
    lower = np.concatenate((np.ones(16), [retained_f - delta - const_f/BASE_FLOPS,
                                          retained_p - delta - const_p/BASE_PARAMS]))
    upper = np.concatenate((np.ones(16), [retained_f - const_f/BASE_FLOPS,
                                          retained_p - const_p/BASE_PARAMS]))
    result = milp(c=objectives, integrality=np.ones(n), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={'time_limit': time_limit, 'mip_rel_gap': 0.001, 'disp': False})
    if result.x is None:
        raise RuntimeError(f'No feasible original-channel allocation: {result.message}')
    chosen = [records[j] for j in np.flatnonzero(result.x > 0.5)]
    if len(chosen) != 16 or sorted(r[0] for r in chosen) != list(range(16)):
        raise RuntimeError('MILP did not return an integer architecture.')
    cfg = [64]
    layer_counts = {}
    for i, a, b, _, _, _ in sorted(chosen):
        cfg.extend((infos[i][3], a, b))
        layer_counts[infos[i][0]+'.conv1'] = a
        layer_counts[infos[i][0]+'.conv2'] = b
    actual = profile_cfg(cfg)
    fd, pd = check_budget(*actual, config)
    report = {'cfg': cfg, 'layer_counts': layer_counts,
              'pruned_flops': actual[0], 'pruned_params': actual[1],
              'flops_drop_percent': fd, 'params_drop_percent': pd,
              'baseline_flops': BASE_FLOPS, 'baseline_params': BASE_PARAMS,
              'solver_status': int(result.status), 'solver_message': result.message,
              'mip_gap': float(getattr(result, 'mip_gap', float('nan'))),
              'retained_arow_score': sum(r[5] for r in chosen),
              'selection': 'maximize sum of retained robust A-row scores subject to both budgets',
              'channels_per_layer': 'top-k unique original DSF-scored output channels',
              'channel_widening': False, 'channel_duplication': False}
    return report
