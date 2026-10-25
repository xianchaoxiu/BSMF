import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import time
from doublesparse import factorize
from utils import *
import os
from models.built_model import *  # 从models文件中导入built_model文件
import re
from select_channel import get_channel_L2

# ------------------------------
# Layer selection and calibration utilities.
# ------------------------------
def _parse_csv_list(s):
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s if str(x).strip()]
    return [x.strip() for x in str(s).split(",") if x.strip()]

def _should_skip_layer(name, module, args):
    # Match layer names and exclusion substrings.
    exclude_layers = set(_parse_csv_list(getattr(args, "dsf_exclude_layers", "")))
    exclude_substr = _parse_csv_list(getattr(args, "dsf_exclude_substr", "downsample"))
    if name in exclude_layers:
        return True
    for ss in exclude_substr:
        if ss and (ss in name):
            return True
    return False

def _tensor_density(x: torch.Tensor) -> float:
    # proportion of non-zero entries
    if x.numel() == 0:
        return 0.0
    return float((x != 0).sum().item()) / float(x.numel())


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _is_main_process(args):
    return int(getattr(args, "rank", 0)) == 0


def _checkpoint_state_dict(model):
    return _unwrap_model(model).state_dict()


def _load_state_dict_into_model(model, state_dict, strict=True):
    return _unwrap_model(model).load_state_dict(state_dict, strict=strict)


def _rel_recon_err(W0_mat: torch.Tensor, W_hat: torch.Tensor) -> float:
    # ||W_hat - W0||_F / ||W0||_F
    num = torch.norm(W_hat - W0_mat, p="fro")
    den = torch.norm(W0_mat, p="fro") + 1e-12
    return float((num / den).item())

@torch.no_grad()
def _bn_calibrate(model, train_loader, device, args, num_batches: int = 200):
    """Update batch-normalization statistics with training batches."""
    if num_batches <= 0:
        return
    was_training = model.training
    model.train()
    it = 0
    for i, (inputs, _) in enumerate(train_loader):
        inputs = inputs.to(device)
        if getattr(args, "half", False):
            inputs = inputs.half()
        _ = model(inputs)
        it += 1
        if it >= num_batches:
            break
    if not was_training:
        model.eval()

def _weight_to_2d(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() == 4:
        return weight.detach().float().view(weight.shape[0], -1)
    elif weight.dim() == 2:
        return weight.detach().float()
    else:
        return weight.detach().float().view(weight.shape[0], -1)


def _init_low_rank_AB(W0_mat: torch.Tensor, rank: int, init_mode: str = "svd"):
    """
    W0_mat: [m, n]
    return:
        A0: [m, r]
        B0: [r, n]
    """
    m, n = W0_mat.shape
    r = min(rank, m, n)

    if init_mode == "svd":
        U, S, Vh = torch.linalg.svd(W0_mat, full_matrices=False)
        Ur = U[:, :r]
        Sr = torch.diag(torch.sqrt(S[:r]))
        Vr = Vh[:r, :]
        A0 = Ur @ Sr
        B0 = Sr @ Vr
        return A0, B0

    raise ValueError(f"Unsupported init_mode: {init_mode}")


def _init_AB_with_factorize_then_svd(W0_mat: torch.Tensor, sparsity: float, rank: int):
    """Initialize low-rank factors by sparse factorization followed by truncated SVD."""
    m, n = W0_mat.shape
    XX = torch.eye(n, device=W0_mat.device, dtype=W0_mat.dtype)

    W_init, _, _ = factorize(W0_mat, XX, sparsity=sparsity, nofinal=True)
    W_init = W_init.to(W0_mat.device, dtype=W0_mat.dtype)

    A0, B0 = _init_low_rank_AB(W_init, rank=rank, init_mode="svd")
    return A0, B0, W_init

def _is_imagenet_resnet50(args) -> bool:
    return (
        str(getattr(args, "model", "")).lower() == "resnet50"
        and str(getattr(args, "dataset", "")).lower() == "imagenet"
    )


def _parse_resnet50_bottleneck_conv(lname: str):
    m = re.search(r"layer(\d+)\.(\d+)\.conv([123])$", lname)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _is_resnet50_imagenet_high_sensitive(lname: str, args) -> bool:
    if not _is_imagenet_resnet50(args):
        return False
    parsed = _parse_resnet50_bottleneck_conv(lname)
    if parsed is None:
        return False
    stage, block, conv = parsed
    if conv not in (1, 2):
        return False
    return (
        block == 0
        or (stage == 4 and block == 1)
    )


def _is_resnet50_imagenet_medium_sensitive(lname: str, args) -> bool:
    if not _is_imagenet_resnet50(args):
        return False
    parsed = _parse_resnet50_bottleneck_conv(lname)
    if parsed is None:
        return False
    stage, block, conv = parsed
    if conv not in (1, 2):
        return False
    if _is_resnet50_imagenet_high_sensitive(lname, args):
        return False
    return (
        (stage == 1 and block in (1, 2))
        or
        (stage == 2 and block == 1)
        or (stage == 3 and block in (1, 2))
    )


def _use_resnet50_layerwise_prune(args) -> bool:
    return (
        _is_imagenet_resnet50(args)
        and bool(getattr(args, "resnet50_layerwise_prune", False))
    )


def _resnet50_layerwise_keep_ratio(lname: str, args):
    parsed = _parse_resnet50_bottleneck_conv(lname)
    if parsed is None:
        return None
    stage, block, conv = parsed
    if conv not in (1, 2):
        return None

    if block == 0:
        keep = float(getattr(args, "resnet50_keep_transition", 0.70))
    elif stage == 1:
        keep = float(getattr(args, "resnet50_keep_stage1", 0.75))
    elif stage == 2:
        keep = float(getattr(args, "resnet50_keep_stage2", 0.58))
    elif stage == 3:
        keep = float(getattr(args, "resnet50_keep_stage3", 0.42))
    elif stage == 4:
        keep = float(getattr(args, "resnet50_keep_stage4", 0.50))
    else:
        keep = 1.0

    if stage == 4 and block == 2:
        keep = max(keep, float(getattr(args, "resnet50_keep_last_block", 0.70)))
    if conv == 2:
        keep += float(getattr(args, "resnet50_conv2_keep_boost", 0.10))

    return min(1.0, max(0.01, keep))


def _resnet50_layerwise_keep_bounds(lname: str, args):
    keep = _resnet50_layerwise_keep_ratio(lname, args)
    if keep is None:
        return None
    margin = max(0.0, float(getattr(args, "resnet50_layerwise_bound_margin", 0.0)))
    min_ratio = max(0.01, keep - margin)
    max_ratio = min(1.0, keep + margin)
    return min_ratio, max(min_ratio, max_ratio)


def _is_high_sensitive_layer(lname: str, args=None) -> bool:
    return any(x in lname for x in [
        'layer1.block_0.conv_bn1.conv',
        'layer2.block_0.conv_bn1.conv',
        'layer3.block_0.conv_bn1.conv',
    ])

def _is_medium_sensitive_layer(lname: str, args=None) -> bool:
    return any(x in lname for x in [
        'layer1.block_1.conv_bn1.conv',
        'layer2.block_1.conv_bn1.conv',
        'layer3.block_1.conv_bn1.conv',
    ])


def _parse_cifar_basicblock_name(lname: str):
    m = re.search(r"layer(\d+)\.block_(\d+)\.conv_bn([12])\.conv", lname)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _is_resnet56_cifar(args) -> bool:
    return (
        str(getattr(args, "model", "")).lower() == "resnet56"
        and str(getattr(args, "dataset", "")).lower() in ("cifar10", "cifar100")
    )


def _use_resnet56_layerwise_bounds(args) -> bool:
    return (
        _is_resnet56_cifar(args)
        and (
            bool(getattr(args, "resnet56_layerwise_prune", False))
            or bool(getattr(args, "resnet56_layerwise_bounds_only", False))
        )
    )


def _use_resnet56_adaptive_protect(args) -> bool:
    return (
        _is_resnet56_cifar(args)
        and (
            bool(getattr(args, "resnet56_adaptive_protect", False))
            or bool(getattr(args, "resnet56_layerwise_prune", False))
        )
    )


def _resnet56_layerwise_keep_ratio(lname: str, args):
    parsed = _parse_cifar_basicblock_name(lname)
    if parsed is None:
        return None

    stage, block, conv = parsed
    if conv != 1:
        return None

    if stage == 1:
        keep = float(getattr(args, "resnet56_keep_stage1", 0.70))
    elif block == 0:
        keep = float(getattr(args, "resnet56_keep_transition", 0.65))
    elif stage == 2:
        keep = float(getattr(args, "resnet56_keep_stage2", 0.40))
    elif stage == 3:
        keep = float(getattr(args, "resnet56_keep_stage3", 0.30))
    else:
        keep = 1.0

    last_block_count = max(0, int(getattr(args, "resnet56_last_block_count", 2)))
    if stage == 3 and last_block_count > 0 and block >= 9 - last_block_count:
        keep = max(keep, float(getattr(args, "resnet56_keep_last_block", 0.48)))

    return min(1.0, max(0.01, keep))


def _resnet56_layerwise_keep_bounds(lname: str, args):
    keep = _resnet56_layerwise_keep_ratio(lname, args)
    if keep is None:
        return None

    parsed = _parse_cifar_basicblock_name(lname)
    margin = max(0.0, float(getattr(args, "resnet56_layerwise_bound_margin", 0.12)))
    min_ratio = max(0.01, keep - margin)
    max_ratio = min(1.0, keep + margin)

    if parsed is not None:
        stage, block, conv = parsed
        if conv == 1:
            if block == 0:
                min_ratio = max(min_ratio, float(getattr(args, "min_keep_sensitive", 0.50)))
                max_ratio = max(max_ratio, min(1.0, keep + margin * 0.75))
            elif block == 1:
                min_ratio = max(min_ratio, float(getattr(args, "min_keep_medium", 0.35)))
            if stage == 1:
                min_ratio = max(min_ratio, 0.50)
            last_block_count = max(0, int(getattr(args, "resnet56_last_block_count", 2)))
            if stage == 3 and last_block_count > 0 and block >= 9 - last_block_count:
                min_ratio = max(min_ratio, float(getattr(args, "resnet56_keep_last_block", 0.48)))

    min_ratio = min(1.0, max(0.01, min_ratio))
    max_ratio = min(1.0, max(min_ratio, max_ratio))
    return min_ratio, max_ratio


def _resnet56_base_floor_ratio(lname: str, args):
    parsed = _parse_cifar_basicblock_name(lname)
    if parsed is None:
        return None

    stage, block, conv = parsed
    if conv != 1:
        return None

    if block == 0:
        floor = float(getattr(args, "min_keep_sensitive", 0.50))
    elif block == 1:
        floor = float(getattr(args, "min_keep_medium", 0.35))
    elif stage == 1:
        floor = float(getattr(args, "min_keep_sensitive", 0.50))
    else:
        floor = float(getattr(args, "min_keep_normal", 0.15))

    last_block_count = max(0, int(getattr(args, "resnet56_last_block_count", 2)))
    if stage == 3 and last_block_count > 0 and block >= 9 - last_block_count:
        floor = max(floor, float(getattr(args, "resnet56_keep_last_block", 0.48)))

    return min(1.0, max(0.01, floor))


def _resnet56_adaptive_limits(lname: str, args):
    parsed = _parse_cifar_basicblock_name(lname)
    if parsed is None:
        return None

    stage, block, conv = parsed
    if conv != 1:
        return None

    min_override = None
    max_override = float(getattr(args, "resnet56_adaptive_max_keep", 1.0))

    if block not in (0, 1):
        if stage == 2:
            min_override = getattr(args, "resnet56_adaptive_min_stage2", None)
            stage_max = getattr(args, "resnet56_adaptive_max_stage2", None)
            if stage_max is not None:
                max_override = float(stage_max)
        elif stage == 3:
            min_override = getattr(args, "resnet56_adaptive_min_stage3", None)
            stage_max = getattr(args, "resnet56_adaptive_max_stage3", None)
            if stage_max is not None:
                max_override = float(stage_max)

    last_block_count = max(0, int(getattr(args, "resnet56_last_block_count", 2)))
    if stage == 3 and last_block_count > 0 and block >= 9 - last_block_count:
        last_max = getattr(args, "resnet56_adaptive_max_last_block", None)
        if last_max is not None:
            max_override = float(last_max)

    if min_override is not None:
        min_override = float(min_override)

    return min_override, min(1.0, max(0.01, float(max_override)))


def _is_structured_prunable_layer(lname: str, args) -> bool:
    mask_scope = str(getattr(args, "dsf_mask_scope", "prunable"))
    if mask_scope == "all":
        return True

    model_name = str(getattr(args, "model", "")).lower()
    if model_name in ["resnet56", "resnet110"]:
        return "conv_bn1.conv" in lname

    if model_name == "resnet50":
        return (
            "downsample" not in lname
            and "conv3" not in lname
            and (
                _use_resnet50_layerwise_prune(args)
                or "layer4.2" not in lname
            )
        )

    return True

def _get_local_lambda_B(lname: str, args) -> float:
    base = float(getattr(args, "lambda_B", 1e-4))

    layerwise_keep = None
    if _use_resnet56_layerwise_bounds(args):
        layerwise_keep = _resnet56_layerwise_keep_ratio(lname, args)

    if _is_high_sensitive_layer(lname, args) or (
        layerwise_keep is not None and layerwise_keep >= 0.66
    ):
        return base * float(getattr(args, "lambda_B_sensitive_scale", 0.50))
    if _is_medium_sensitive_layer(lname, args) or (
        layerwise_keep is not None and layerwise_keep >= 0.50
    ):
        return base * float(getattr(args, "lambda_B_medium_scale", 0.75))
    return base

def _get_keep_bounds(lname: str, num_channels: int, args):
    layerwise_keep = None
    layerwise_bounds = None
    if _use_resnet56_adaptive_protect(args):
        base_floor = _resnet56_base_floor_ratio(lname, args)
        if base_floor is not None:
            # DSF hard masks are also used by the final structured-pruning pass.
            # Keep sensitive ResNet-56 layers above their protection floor here,
            # so they are not damaged before the final A-row-norm repair step.
            min_ratio = base_floor
            limits = _resnet56_adaptive_limits(lname, args)
            if limits is not None:
                min_override, _max_override = limits
                if min_override is not None:
                    min_ratio = max(min_ratio, float(min_override))
            max_ratio = 1.0
            floor = 2
            min_keep = max(floor, int(math.ceil(num_channels * min_ratio)))
            min_keep = min(num_channels, max(1, min_keep))
            if max_ratio >= 1.0:
                max_keep = num_channels
            else:
                max_keep = int(math.floor(num_channels * max_ratio))
                max_keep = min(num_channels, max(min_keep, max_keep))
            return min_keep, max_keep
    if _use_resnet56_layerwise_bounds(args):
        layerwise_bounds = _resnet56_layerwise_keep_bounds(lname, args)
    if _use_resnet50_layerwise_prune(args):
        layerwise_bounds = _resnet50_layerwise_keep_bounds(lname, args)

    if layerwise_bounds is not None:
        min_ratio, max_ratio = layerwise_bounds
        floor = 4
    elif layerwise_keep is not None:
        min_ratio = layerwise_keep
        max_ratio = layerwise_keep
        floor = 4
    elif _is_high_sensitive_layer(lname, args):
        min_ratio = float(getattr(args, "min_keep_sensitive", 0.50))
        max_ratio = float(getattr(args, "max_keep_sensitive", 1.0))
        floor = 4
    elif _is_medium_sensitive_layer(lname, args):
        min_ratio = float(getattr(args, "min_keep_medium", 0.35))
        max_ratio = float(getattr(args, "max_keep_medium", 1.0))
        floor = 4
    else:
        min_ratio = float(getattr(args, "min_keep_normal", 0.15))
        max_ratio = float(getattr(args, "max_keep_normal", 1.0))
        floor = 2

    min_keep = max(floor, int(math.ceil(num_channels * min_ratio)))
    min_keep = min(num_channels, max(1, min_keep))

    if max_ratio >= 1.0:
        max_keep = num_channels
    else:
        max_keep = int(math.floor(num_channels * max_ratio))
        max_keep = min(num_channels, max(min_keep, max_keep))

    return min_keep, max_keep

def _repair_mask_with_keep_bounds(score: torch.Tensor, mask: torch.Tensor, lname: str, args):
    score = score.detach().float()
    mask = mask.detach().float()
    min_keep, max_keep = _get_keep_bounds(lname, int(score.numel()), args)
    remain = int(mask.sum().item())
    repaired = []

    if remain < min_keep:
        _, keep_idx = torch.topk(score, k=min_keep, largest=True)
        new_mask = torch.zeros_like(mask)
        new_mask[keep_idx] = 1.0
        mask = new_mask
        remain = min_keep
        repaired.append(f"min->{min_keep}")

    if remain > max_keep:
        _, keep_idx = torch.topk(score, k=max_keep, largest=True)
        new_mask = torch.zeros_like(mask)
        new_mask[keep_idx] = 1.0
        mask = new_mask
        remain = max_keep
        repaired.append(f"max->{max_keep}")

    return mask, remain, ",".join(repaired) if repaired else "none"

def _get_local_rank_ratio(lname: str, args) -> float:
    base = float(getattr(args, "dsf_rank_ratio", 0.6))
    if _is_high_sensitive_layer(lname, args):
        return max(base, 0.80)
    if _is_medium_sensitive_layer(lname, args):
        return max(base, 0.70)
    return base

def _get_local_rank_energy(lname: str, args) -> float:
    base = float(getattr(args, "dsf_rank_energy", 0.90))
    if _is_high_sensitive_layer(lname, args):
        return min(0.99, max(base, 0.94))
    if _is_medium_sensitive_layer(lname, args):
        return min(0.99, max(base, 0.92))
    return min(0.99, max(0.50, base))

def _rank_from_spectrum(svals: torch.Tensor, target_energy: float) -> int:
    if svals.numel() == 0:
        return 1
    energy = svals.detach().float().pow(2)
    total = energy.sum()
    if float(total.item()) <= 1e-12:
        return 1
    cumulative = torch.cumsum(energy, dim=0) / total
    hits = torch.nonzero(cumulative >= target_energy, as_tuple=False)
    if hits.numel() == 0:
        return int(svals.numel())
    return int(hits[0].item()) + 1

def _energy_at_rank(svals: torch.Tensor, rank: int) -> float:
    if svals.numel() == 0:
        return 0.0
    energy = svals.detach().float().pow(2)
    total = energy.sum()
    if float(total.item()) <= 1e-12:
        return 1.0
    rank = min(max(1, int(rank)), int(svals.numel()))
    return float((energy[:rank].sum() / total).item())

@torch.no_grad()
def _resolve_low_rank(lname: str, W0_mat: torch.Tensor, args):
    m, n = W0_mat.shape
    rank_mode = str(getattr(args, "dsf_rank_mode", "auto")).lower()
    local_rank_ratio = _get_local_rank_ratio(lname, args)
    rank_param_frac = float(getattr(args, "dsf_rank_param_frac", 0.90))
    min_rank = max(1, int(getattr(args, "dsf_min_rank", 1)))
    target_energy = None
    retained_energy = None

    if rank_mode == "auto":
        target_energy = _get_local_rank_energy(lname, args)
        svals = torch.linalg.svdvals(W0_mat.detach().float())
        requested_rank = _rank_from_spectrum(svals, target_energy)
        rank_source = "auto_spectrum"
    else:
        requested_rank = max(1, int(min(m, n) * local_rank_ratio))
        rank_source = "fixed_ratio"
        svals = None

    if rank_param_frac > 0.0 and rank_param_frac < 1.0:
        max_rank_by_params = int(math.floor((rank_param_frac * m * n) / max(m + n, 1)))
        max_rank_by_params = max(min_rank, max_rank_by_params)
        rank = min(requested_rank, max_rank_by_params)
    else:
        max_rank_by_params = None
        rank = requested_rank

    rank = min(max(rank, min_rank), min(m, n))
    if svals is not None:
        retained_energy = _energy_at_rank(svals, rank)
    orig_params = int(m * n)
    factor_params = int(rank * (m + n))
    factor_param_frac = factor_params / max(orig_params, 1)
    return (
        rank,
        requested_rank,
        rank_source,
        local_rank_ratio,
        target_energy,
        retained_energy,
        max_rank_by_params,
        factor_params,
        factor_param_frac,
    )

def _get_local_lambda_A(lname: str, args) -> float:
    base = float(getattr(args, "lambda_A", 1e-3))
    if _is_high_sensitive_layer(lname, args):
        return base * 0.35
    if _is_medium_sensitive_layer(lname, args):
        return base * 0.60
    return base

def _normalize_score_01(score: torch.Tensor) -> torch.Tensor:
    score = score.detach().float()
    smin = score.min()
    smax = score.max()
    if float((smax - smin).abs()) < 1e-12:
        return torch.ones_like(score)
    return (score - smin) / (smax - smin + 1e-12)

def _normalize_score_01_robust(score: torch.Tensor, q_low: float = 0.05, q_high: float = 0.95) -> torch.Tensor:
    score = score.detach().float()
    if score.numel() == 0:
        return score
    lo = torch.quantile(score, q_low)
    hi = torch.quantile(score, q_high)
    if float((hi - lo).abs()) < 1e-12:
        return torch.ones_like(score)
    return ((score - lo) / (hi - lo + 1e-12)).clamp(0.0, 1.0)


# 用途：根据 A 的行范数构造 soft-gated effective weight，即 Weff = diag(g)AB


def _build_channel_score_variants(raw_score, W_comp, args):
    raw_score = raw_score.detach().float()
    a_row_robust = _normalize_score_01_robust(raw_score)
    aib_score = torch.linalg.norm(W_comp.detach().float(), ord=2, dim=1)
    aib_robust = _normalize_score_01_robust(aib_score)

    alpha = min(1.0, max(0.0, float(getattr(args, "a_row_aib_alpha", 0.85))))
    beta = max(0.0, float(getattr(args, "a_row_aib_tie_beta", 0.02)))

    variants = {
        "raw": raw_score,
        "robust": a_row_robust,
        "aib_mix": alpha * a_row_robust + (1.0 - alpha) * aib_robust,
        "aib_tie": a_row_robust + beta * aib_robust,
        "random": torch.rand_like(a_row_robust),
    }
    return variants

def compress_model(train_loader, model, criterion, optimizer,
                   epochs, device, args, val_loader,
                   sparsity=0.2):
    # DSF: factorization, ADMM shrinkage, weight reconstruction, and finetuning.
    #
    # one-shot 分解得到 A0,B0（只做权重重构 W0≈A0B0）
    #     -> 仅对 A/B 做 ADMM shrink（p<1 + M 抑制），跑 K 轮
    #     -> 乘回 W 并写回模型
    #     -> 用交叉熵 finetune 拉回精度（epochs 轮）
    #
    # 参数:
    #   - epochs: finetuning epochs after factorization and shrinkage.
    #   - sparsity: 传给 doublesparse.factorize 的目标稀疏率

    model.to(device)
    model_for_layers = _unwrap_model(model)


    save_dir = os.path.join(getattr(args, "save_dir", "save_temp"), "compress_model")
    if _is_main_process(args):
        os.makedirs(save_dir, exist_ok=True)


    # 裸 AB 重构权重：Wcomp = AB
    # soft-gated 有效权重：Weff = diag(g)AB

    # ------------------------------------------------------------------
    # 0) 建立每层状态：保存 W0，并准备存 A/B (这里 A,B 就对应 Z_A,Z_B)
    # ------------------------------------------------------------------
    layer_state = {}
    for name, module in model_for_layers.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 跳过对 ACC 更敏感、但对 FLOPs 收益不大的层
            if (
                    name in ["conv1", "fc", "linear", "conv_bn.conv"]
                    or isinstance(module, nn.Linear)
                    or _should_skip_layer(name, module, args)
            ):
                continue

            w_init = module.weight.data.clone()

            layer_state[name] = {
                'module': module,
                'W0': w_init.to(device),  # 固定的原始权重 W0
                'A': None,
                'B': None,
                'U_A': None,
                'U_B': None,
            }


    print("===> Start DSF: one-shot factorize -> ADMM shrink(A/B) -> writeback -> finetune")
    print("Compressible layers:", list(layer_state.keys()))

    # ------------------------------------------------------------------
    # Stage 1: one-shot factorization（得到 A0,B0，只做一次）
    # ------------------------------------------------------------------
    print("===> Stage1: one-shot factorization (W0 ≈ A0 B0)")
    init_mode = str(getattr(args, "dsf_init", "svd"))
    low_rank_orig_params_total = 0
    low_rank_factor_params_total = 0
    low_rank_capped_layers = 0

    for lname, st in layer_state.items():
        module = st['module']
        W0 = st['W0']
        W0_mat = _weight_to_2d(W0).to(device)

        m, n = W0_mat.shape
        (
            rank,
            requested_rank,
            rank_source,
            local_rank_ratio,
            target_energy,
            retained_energy,
            max_rank_by_params,
            factor_params,
            factor_param_frac,
        ) = _resolve_low_rank(lname, W0_mat, args)
        orig_params = int(m * n)
        low_rank_orig_params_total += orig_params
        low_rank_factor_params_total += factor_params
        if max_rank_by_params is not None and rank < requested_rank:
            low_rank_capped_layers += 1

        if init_mode == "svd":
            A0, B0 = _init_low_rank_AB(W0_mat, rank=rank, init_mode="svd")
        elif init_mode == "factorize_svd":
            A0, B0, _ = _init_AB_with_factorize_then_svd(
                W0_mat, sparsity=sparsity, rank=rank
            )
        else:
            raise ValueError(f"Unknown dsf_init: {init_mode}")

        st['A'] = A0.clone()
        st['B'] = B0.clone()
        st['U_A'] = torch.zeros_like(A0)
        st['U_B'] = torch.zeros_like(B0)

        with torch.no_grad():
            W_hat = A0 @ B0
            rel_err = _rel_recon_err(W0_mat, W_hat)
            densA = _tensor_density(A0)
            densB = _tensor_density(B0)
            cap_text = f" cap={max_rank_by_params}" if max_rank_by_params is not None else ""
            source_text = f"source={rank_source}"
            if rank_source == "fixed_ratio":
                source_text += f" local_rank_ratio={local_rank_ratio:.2f}"
            energy_text = ""
            if target_energy is not None and retained_energy is not None:
                energy_text = f" energy={retained_energy:.3f}/{target_energy:.3f}"
            print(
                f"[Stage1-init] {lname}: rank={rank}/{min(m, n)} requested={requested_rank} "
                f"{source_text}{energy_text}{cap_text} "
                f"lowrank_params={factor_params}/{orig_params} ({factor_param_frac:.3f}) "
                f"rel_err={rel_err:.6f} densA={densA:.4f} densB={densB:.4f} "
                f"A{tuple(A0.shape)} B{tuple(B0.shape)}"
            )

    if low_rank_orig_params_total > 0:
        low_rank_frac = low_rank_factor_params_total / float(low_rank_orig_params_total)
        print(
            f"[Stage1-summary] lowrank_factor_params={low_rank_factor_params_total}/"
            f"{low_rank_orig_params_total} ({low_rank_frac:.3f}), "
            f"param_drop={(1.0 - low_rank_frac) * 100.0:.2f}%, "
            f"rank_capped_layers={low_rank_capped_layers}/{len(layer_state)}"
        )

    # Validate the reconstructed AB weights.
    for lname, st in layer_state.items():
        module = st['module']
        A, B = st.get('A'), st.get('B')
        if A is None or B is None:
            print(f"[WARN] {lname}: A/B None, skip writeback")
            continue
        W_comp0 = A @ B
        module.weight.data.copy_(W_comp0.view_as(module.weight.data))

    print("===> validate after factorize writeback (A0@B0)")
    # Calibrate BN statistics after weight reconstruction.
    bn_calib_batches = int(getattr(args, "bn_calib_batches", 200))
    _bn_calibrate(model, train_loader, device, args, num_batches=bn_calib_batches)
    factorize_prec1, factorize_prec5 = validate(val_loader, model, criterion, args, device)

    # 再恢复原始 W0
    for lname, st in layer_state.items():
        st['module'].weight.data.copy_(st['W0'])

    # ------------------------------------------------------------------
    # Stage 2: 稀疏优化
    # ------------------------------------------------------------------
    #稀疏参数设置
    shrink_iters = int(getattr(args, 'shrink_iters', 5))
    rho = float(getattr(args, 'rho', 1e-3))
    p_A = float(getattr(args, 'p', 0.5))
    p_B = float(getattr(args, 'q', 0.5))
    lambda_A = float(getattr(args, 'lambda_A', 3e-3))
    lambda_B = float(getattr(args, 'lambda_B', 1e-4))

    print(f"===> Stage2: low-rank ADMM shrink on A/B")

    # For the ResNet-50/ImageNet path, optionally select the best of the
    # shrink snapshots by validation accuracy.  The default remains disabled
    # for all other models and datasets.
    select_shrink_by_val = bool(
        getattr(args, 'resnet50_shrink_select_by_val', False)
        and _is_imagenet_resnet50(args)
    )
    best_shrink_prec1 = -1.0
    best_shrink_state = None
    shrink_probe_batches = int(getattr(args, 'resnet50_shrink_probe_bn_batches', 80))

    for k in range(shrink_iters):
        for lname, st in layer_state.items():
            W0 = _weight_to_2d(st['W0']).to(device)  # [m, n]

            Z_A = st['A']  # [m, r]
            Z_B = st['B']  # [r, n]
            U_A = st['U_A']  # [m, r]
            U_B = st['U_B']  # [r, n]

            # ----- A-step -----
            B_use = Z_B
            BBt = B_use @ B_use.t()  # [r, r]
            I_r = torch.eye(BBt.shape[0], device=W0.device, dtype=W0.dtype)
            RHS_A = W0 @ B_use.t() + rho * (Z_A - U_A)  # [m, r]
            A = RHS_A @ torch.linalg.inv(BBt + rho * I_r)

            V_A = A + U_A
            local_lambda_A = _get_local_lambda_A(lname, args)
            Z_A_new = prox_group_l2p_pow(
                V_A,
                tau=(local_lambda_A / rho),
                p=p_A,
                group="row"  # A 的行对应输出通道
            )
            U_A_new = U_A + (A - Z_A_new)

            # ----- B-step -----
            AtA = Z_A_new.t() @ Z_A_new  # [r, r]
            I_r = torch.eye(AtA.shape[0], device=W0.device, dtype=W0.dtype)
            RHS_B = Z_A_new.t() @ W0 + rho * (Z_B - U_B)  # [r, n]
            B = torch.linalg.solve(AtA + rho * I_r, RHS_B)

            V_B = B + U_B
            local_lambda_B = _get_local_lambda_B(lname, args)
            Z_B_new = prox_group_l2p_pow(
                V_B,
                tau=float(local_lambda_B / rho),
                p=p_B,
                group="col"
            )
            U_B_new = U_B + (B - Z_B_new)

            st['A'] = Z_A_new.detach()
            st['B'] = Z_B_new.detach()
            st['U_A'] = U_A_new.detach()
            st['U_B'] = U_B_new.detach()

        if (k + 1) % max(1, int(getattr(args, 'shrink_print_freq', 1))) == 0:
            small_thr = float(getattr(args, "group_small_thr", 5e-2))

            A_mean_all, A_med_all, A_small_all = [], [], []
            B_mean_all, B_med_all, B_small_all = [], [], []

            for _lname, _st in layer_state.items():
                ss = _group_shrink_stats(_st["A"], _st["B"], small_thr=small_thr)
                A_mean_all.append(ss["A_row_mean"])
                A_med_all.append(ss["A_row_med"])
                A_small_all.append(ss["A_row_small"])
                B_mean_all.append(ss["B_col_mean"])
                B_med_all.append(ss["B_col_med"])
                B_small_all.append(ss["B_col_small"])

            print(
                f"[shrink {k+1}/{shrink_iters}] "
                f"A_row_mean={np.mean(A_mean_all):.4e}  "
                f"A_row_med={np.mean(A_med_all):.4e}  "
                f"A_row_small@{small_thr:.0e}={np.mean(A_small_all):.3f}  |  "
                f"B_col_mean={np.mean(B_mean_all):.4e}  "
                f"B_col_med={np.mean(B_med_all):.4e}  "
                f"B_col_small@{small_thr:.0e}={np.mean(B_small_all):.3f}"
        )

        if select_shrink_by_val:
            # Evaluate each ADMM snapshot after fresh BN calibration.  Keep
            # CPU copies so later iterations cannot overwrite the candidate.
            for _lname, _st in layer_state.items():
                _weight = (_st['A'] @ _st['B']).view_as(_st['module'].weight.data)
                _st['module'].weight.data.copy_(_weight)
            _bn_calibrate(
                model, train_loader, device, args,
                num_batches=shrink_probe_batches,
            )
            _shrink_prec1, _shrink_prec5 = validate(
                val_loader, model, criterion, args, device
            )
            print(
                f"[shrink-select] iter={k + 1} "
                f"Prec@1={float(_shrink_prec1):.4f} Prec@5={float(_shrink_prec5):.4f}"
            )
            if float(_shrink_prec1) > best_shrink_prec1:
                best_shrink_prec1 = float(_shrink_prec1)
                best_shrink_state = {
                    _lname: {
                        'A': _st['A'].detach().clone(),
                        'B': _st['B'].detach().clone(),
                    }
                    for _lname, _st in layer_state.items()
                }

    if select_shrink_by_val and best_shrink_state is not None:
        for _lname, _st in layer_state.items():
            _st['A'] = best_shrink_state[_lname]['A']
            _st['B'] = best_shrink_state[_lname]['B']
        print(f"===> Selected best DSF shrink snapshot: Prec@1={best_shrink_prec1:.4f}")

    # Stage 3: write back compressed weights and export channel scores
    print("===> Stage3: write back compressed weights and export channel scores")

    channel_scores = {}
    channel_score_variants = {
        "raw": {},
        "robust": {},
        "aib_mix": {},
        "aib_tie": {},
        "random": {},
    }
    weff_score_dict = {}
    weff_mask_dict = {}

    for lname, st in layer_state.items():
        module = st['module']
        A = st['A']
        B = st['B']

        # 1) 裸重构权重：W_comp = AB
        W_comp = A @ B  # [C_out, C_in * k * k]

        module.weight.data.copy_(W_comp.view_as(module.weight.data))

        # 2) A 的行范数就是输出通道重要性
        raw_score = torch.linalg.norm(A, ord=2, dim=1)  # [C_out]


        # 用途：收集 A 行范数、裸 AB 权重和 soft-gated Weff 权重

        if isinstance(module, nn.Conv2d):
            # 结构化剪枝的主通道分数来自 A 的行 L2 范数。
            # A.shape = [C_out, rank]，第 i 行严格对应第 i 个输出通道。
            # AiB 变体只用于评分消融：||A_i B||_2 衡量该通道重构权重行的贡献。
            score_variants = _build_channel_score_variants(raw_score, W_comp, args)
            for variant_name, variant_score in score_variants.items():
                channel_score_variants[variant_name][lname] = variant_score.detach().cpu()

            score_mode = str(getattr(args, "a_row_norm_score", "robust")).lower()
            if score_mode not in score_variants:
                print(f"[WARN] unknown A-row score mode {score_mode}, fallback to robust")
                score_mode = "robust"
            base_score = score_variants[score_mode].detach().float()

            # soft Weff 只用于展示“组稀疏诱导效果”


            # 后续剪枝也用同一个 A-row-norm 分数；hard mask 默认只覆盖最终会被结构化剪枝的层
            channel_scores[lname] = base_score.detach().cpu()
            if _is_structured_prunable_layer(lname, args):
                weff_score_dict[lname] = base_score.detach().cpu()
            print(
                f"[A-Row-Norm] {lname}: mode={score_mode} score_shape={tuple(base_score.shape)} "
                f"raw_min={float(raw_score.min().item()):.6e} "
                f"raw_median={float(raw_score.median().item()):.6e} "
                f"raw_max={float(raw_score.max().item()):.6e} "
                f"aib_alpha={float(getattr(args, 'a_row_aib_alpha', 0.85)):.3f} "
                f"aib_tie_beta={float(getattr(args, 'a_row_aib_tie_beta', 0.02)):.3f}"
            )

    # ===== 用和 Weff 相同的分数生成全局 hard mask =====
    all_scores = []
    for lname, score in weff_score_dict.items():
        all_scores.append(score.reshape(-1))

    if len(all_scores) > 0:
        all_scores = torch.cat(all_scores, dim=0)
        y, _ = torch.sort(all_scores)
        thre_index = min(int(all_scores.numel() * args.ratio), all_scores.numel() - 1)
        thre = y[thre_index]

        for lname, score in weff_score_dict.items():
            if _use_resnet50_layerwise_prune(args):
                mask = torch.ones_like(score)
            else:
                mask = score.gt(thre).float()
            mask, remain, repair = _repair_mask_with_keep_bounds(score, mask, lname, args)
            weff_mask_dict[lname] = mask.cpu()
            print(
                f"[Weff-Mask] {lname}: keep={remain}/{score.numel()} "
                f"({remain / max(score.numel(), 1):.3f}) repair={repair}"
            )

        print(
            f"[Weff-Mask] scope={getattr(args, 'dsf_mask_scope', 'prunable')} "
            f"layers={len(weff_score_dict)} global threshold = {float(thre):.6f}"
        )

    bn_calib_batches = int(getattr(args, "bn_calib_batches", 200))
    _bn_calibrate(model, train_loader, device, args, num_batches=bn_calib_batches)

    print("===> validate after compression (before finetune)")
    prec1_comp, prec5_comp = validate(val_loader, model, criterion, args, device)
    min_comp_prec1 = getattr(args, "resnet50_min_compressed_prec1", None)
    if _is_imagenet_resnet50(args) and min_comp_prec1 is not None:
        min_comp_prec1 = float(min_comp_prec1)
        if float(prec1_comp) < min_comp_prec1:
            raise RuntimeError(
                f"ResNet50/ImageNet compressed Prec@1={float(prec1_comp):.3f} "
                f"is below --resnet50-min-compressed-prec1={min_comp_prec1:.3f}; "
                "abort before long fine-tuning."
            )


    if _is_main_process(args):
        dataset_name = str(args.dataset).upper().replace("_", "-")
        model_name = str(args.model).replace("resnet", "ResNet-").replace("vgg", "VGG-")


    # ------------------------------------------------------------------
    # Stage 4: finetune（交叉熵）
    # ------------------------------------------------------------------
    print("===> Stage4: finetune compressed model, epochs =", epochs)

    best_ft_prec1 = -1.0
    best_ft_state = None

    dsf_warm_epochs = int(getattr(args, 'dsf_warm_epochs', 0))
    dsf_warm_lr = float(getattr(args, 'dsf_warm_lr', 0.0))
    dsf_peak_lr = float(args.dsf_lr)
    dsf_num_iter = max(1, len(train_loader))
    dsf_total_steps = max(1, int(epochs) * dsf_num_iter)
    dsf_warm_steps = max(0, min(dsf_total_steps, dsf_warm_epochs * dsf_num_iter))
    if dsf_warm_steps > 0:
        dsf_warm_lr = max(0.0, min(dsf_peak_lr, dsf_warm_lr))

    def _dsf_lr_at_step(global_step):
        if dsf_warm_steps > 0 and global_step < dsf_warm_steps:
            # The first optimizer step is deliberately small, then rises
            # linearly to the requested peak at the warmup boundary.
            alpha = float(global_step + 1) / float(dsf_warm_steps)
            return dsf_warm_lr + (dsf_peak_lr - dsf_warm_lr) * alpha
        if dsf_warm_steps > 0:
            post_steps = max(1, dsf_total_steps - dsf_warm_steps)
            progress = float(global_step - dsf_warm_steps) / float(post_steps)
            progress = min(1.0, max(0.0, progress))
            return dsf_peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        # Use epoch-wise cosine decay when warmup is disabled.
        epoch = min(int(global_step // dsf_num_iter), max(0, int(epochs) - 1))
        return dsf_peak_lr * 0.5 * (
            1.0 + math.cos(math.pi * epoch / max(1, int(epochs)))
        )

    for epoch in range(epochs):
        if hasattr(getattr(train_loader, 'sampler', None), 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Use step-wise warmup and cosine decay when enabled;
        # otherwise use epoch-wise cosine decay.
        current_lr = _dsf_lr_at_step(epoch * dsf_num_iter)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        print(
            f"===> FT Epoch {epoch:3d}: lr = {current_lr:.8f} "
            f"(warm_epochs={dsf_warm_epochs}, warm_lr={dsf_warm_lr:.8f})"
        )

        model.train()

        # 前 3 个 epoch 不冻结 BN，先让统计量适配写回后的权重
        if epoch >= 3:
            freeze_bn(model)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # Use the configured label smoothing during finetuning.
        dsf_label_smoothing = float(getattr(args, "dsf_label_smoothing", 0.05))
        train_criterion = nn.CrossEntropyLoss(label_smoothing=dsf_label_smoothing).to(device)

        end = time.time()
        for i, (inputs, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            if dsf_warm_steps > 0:
                current_lr = _dsf_lr_at_step(epoch * dsf_num_iter + i)
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr

            inputs = inputs.to(device)
            target = target.to(device)
            if getattr(args, 'half', False):
                inputs = inputs.half()

            output = model(inputs)
            loss = train_criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print('FT Epoch: [{0}][{1}/{2}]\t'
                      'LR {lr:.8f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader),
                    lr=current_lr,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1))

        # 每轮结束后做一个短 BN 校准
        _bn_calibrate(model, train_loader, device, args, num_batches=50)

        prec1, prec5 = validate(val_loader, model, criterion, args, device)

        is_best_ft = float(prec1) > best_ft_prec1
        if is_best_ft:
            best_ft_prec1 = float(prec1)
            best_ft_state = {
                k: v.detach().cpu().clone()
                for k, v in _checkpoint_state_dict(model).items()
            }

        if _is_main_process(args):
            payload = {
                'epoch': epoch,
                'model_state_dict': _checkpoint_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'prec1': prec1,
                'prec5': prec5,
                'channel_scores': channel_scores,
                'channel_score_variants': channel_score_variants,
                'weff_mask_dict': weff_mask_dict,
                'best_prec1': best_ft_prec1,
            }
            checkpoint_stem = (
                f'{args.model}_dsf_shrink_lA{lambda_A}_lB{lambda_B}_rho{rho}'
            )
            latest_path = os.path.join(save_dir, f'{checkpoint_stem}_latest.pth')
            torch.save(payload, latest_path)
            if is_best_ft:
                torch.save(payload, os.path.join(save_dir, f'{checkpoint_stem}_best.pth'))
            if should_save_epoch_checkpoint(
                epoch, epochs, getattr(args, 'save_every', 10)
            ):
                save_path = os.path.join(save_dir, f'{checkpoint_stem}_ft{epoch}.pth')
                torch.save(payload, save_path)
                print(
                    f"===> FT Epoch {epoch}: prec1={prec1:.4f}, prec5={prec5:.4f}, "
                    f"saved milestone {save_path}"
                )
            else:
                print(
                    f"===> FT Epoch {epoch}: prec1={prec1:.4f}, prec5={prec5:.4f}, "
                    f"updated {latest_path}"
                )

    if best_ft_state is not None:
        _load_state_dict_into_model(model, best_ft_state, strict=True)
        print(f"===> Load best DSF finetuned checkpoint: {best_ft_prec1:.4f}")

    return model, channel_scores, weff_mask_dict

# 冻结 BN 层的统计量
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()  # 固定 BN 层的统计量，不进行更新

def prox_scalar_lp(a: torch.Tensor, tau: float, p: float) -> torch.Tensor:
    # Scalar proximal operator.
    eps = 1e-12
    a = torch.clamp(a, min=eps)

    if abs(p - 0.0) < 1e-6:
        thresh = math.sqrt(2 * tau)
        return torch.where(a > thresh, a, torch.zeros_like(a))

    elif abs(p - 0.5) < 1e-6:
        t = (3 * tau / 2) ** (2 / 3)
        mask = a > t
        res = torch.zeros_like(a)
        if mask.any():
            aa = a[mask]
            phi = torch.acos(torch.clamp(tau / 4 * (aa / 3).pow(-1.5), -1 + 1e-7, 1 - 1e-7))
            res[mask] = (2 / 3) * aa * (1 + torch.cos(2 * math.pi / 3 - 2 * phi / 3))
        return res

    elif abs(p - 2 / 3) < 1e-6:
        c = (2 * tau * (1 - p)) ** (1 / (2 - p))
        return torch.relu(a - c * (a ** (p - 1)))

    else:
        # Approximate the general case with a fixed number of iterations.
        x = a.clone()
        for _ in range(3):
            x = torch.relu(a - tau * p * x.pow(p - 1))
        return x


def prox_group_l2p_pow(V: torch.Tensor, tau: float, p: float, group: str) -> torch.Tensor:  ##组 prox + 方向缩放
    #solve: argmin_Z tau * ||Z||_{2,p}^p + 0.5 * ||Z - V||_F^2
    #group:
    # "row": 每行一个组（对 B 推荐）
    # "col": 每列一个组（对 A 推荐）
    eps = 1e-12

    if group == "col":
        Vt = V.t()
        Zt = prox_group_l2p_pow(Vt, tau, p, group="row")
        return Zt.t()

    # group == "row"
    # 每行的L2范数
    r = torch.linalg.norm(V, ord=2, dim=1)          # (G,)
    # 标量prox：prox_{tau |.|^p}(r)
    s = prox_scalar_lp(r, tau, p)                   # (G,)
    coeff = s / (r + eps)                           # (G,)
    return V * coeff.unsqueeze(1)

def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """用 'layer1.0.conv1' 这种 dotted name 替换子模块"""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)

def _group_shrink_stats(A: torch.Tensor, B: torch.Tensor, small_thr: float = 5e-2):
    A_row = torch.linalg.norm(A, ord=2, dim=1)
    B_col = torch.linalg.norm(B, ord=2, dim=0)

    return {
        "A_row_mean": A_row.mean().item(),
        "A_row_med": A_row.median().item(),
        "A_row_small": (A_row < small_thr).float().mean().item(),
        "B_col_mean": B_col.mean().item(),
        "B_col_med": B_col.median().item(),
        "B_col_small": (B_col < small_thr).float().mean().item(),
    }


def _collect_group_shrink_summary(layer_state, small_thr: float = 5e-2):
    buckets = {
        "A_row_mean": [],
        "A_row_med": [],
        "A_row_small": [],
        "B_col_mean": [],
        "B_col_med": [],
        "B_col_small": [],
    }
    for _lname, _st in layer_state.items():
        ss = _group_shrink_stats(_st["A"], _st["B"], small_thr=small_thr)
        for key in buckets:
            buckets[key].append(ss[key])
    return {
        key: float(np.mean(values)) if values else 0.0
        for key, values in buckets.items()
    }


def _clone_layer_ab_state(layer_state):
    return {
        lname: {
            "A": st["A"].detach().clone(),
            "B": st["B"].detach().clone(),
        }
        for lname, st in layer_state.items()
    }


def _write_ab_weights_to_model(layer_state):
    for _lname, st in layer_state.items():
        A = st.get("A")
        B = st.get("B")
        module = st.get("module")
        if A is None or B is None or module is None:
            continue
        W_comp = A @ B
        module.weight.data.copy_(W_comp.view_as(module.weight.data))


def _restore_layer_ab_state(layer_state, snapshot):
    for lname, saved in snapshot.items():
        if lname not in layer_state:
            continue
        layer_state[lname]["A"] = saved["A"].detach().clone()
        layer_state[lname]["B"] = saved["B"].detach().clone()
