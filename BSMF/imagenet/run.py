#!/usr/bin/env python3
"""Run the ResNet-50 / ImageNet-1K main experiment, one GPU at a time."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BEST_NAME = 'resnet50_dsf_shrink_lA0.0012_lB8e-05_rho0.01_best.pth'


def experiment_defaults():
    """Default parameters for the ImageNet pipeline."""
    return {'flops_drop_percent': 50,
     'params_drop_percent': 49,
     'tolerance_pp': 0.05,
     'baseline_flops': 4133742592,
     'baseline_params': 25557032,
     'max_flops': 2066871296.0,
     'max_params': 13034086.32,
     'seed': 10,
     'batch_size': 256,
     'ema': False,
     'kd': False,
     'swa': False,
     'channel_widening': False,
     'start_from': 'torchvision ResNet50 IMAGENET1K_V1',
     'dsf': {'p': 1.0,
             'q': 0.875,
             'lambda_A': 0.0012,
             'lambda_B': 8e-05,
             'rho': 0.01,
             'epochs': 25,
             'lr': 0.0008,
             'momentum': 0.9,
             'weight_decay': 5e-05,
             'label_smoothing': 0.03},
     'finetune': {'epochs': 180,
                  'lr': 0.025,
                  'warmup_lr': 0.0025,
                  'warmup_epochs': 5,
                  'min_lr': 1e-06,
                  'momentum': 0.9,
                  'nesterov': True,
                  'weight_decay': 5e-05,
                  'norm_and_bias_weight_decay': 0,
                  'label_smoothing': 0.05,
                  'clean_tail_epochs': 20,
                  'random_augment': [2, 5],
                  'random_erasing': 0.1,
                  'gradient_clip_norm': 5.0}}


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def dsf_command(run, config, data_dir, workers):
    # Construct the ResNet-50 DSF training command.
    return [sys.executable, '-u', str(HERE / 'source/main.py'),
            '--device', '0', '--model', 'resnet50', '--dataset', 'imagenet',
            '--data-dir', str(data_dir), '--pretrained', '--save-eval-baseline',
            '--epochs', '0', '--fine-epochs', '0', '--batch-size', '256',
            '--workers', str(workers), '--seed', str(config['seed']),
            '--ratio', '0.60', '--p', '1.0', '--q', '0.875',
            '--momentum', '0.9', '--weight-decay', '1e-4', '--dsf',
            '--compressed-epoch', '25', '--dsf-lr', '8e-4',
            '--dsf-momentum', '0.9', '--dsf-weight-decay', '5e-5',
            '--dsf-label-smoothing', '0.03', '--shrink-iters', '3',
            '--lambda_A', '0.0012', '--lambda_B', '8e-05', '--rho', '0.01',
            '--dsf-rank-mode', 'auto', '--dsf-rank-energy', '0.95',
            '--dsf-rank-param-frac', '0.95', '--dsf-min-rank', '1',
            '--dsf-init', 'svd', '--resnet50-shrink-select-by-val',
            '--resnet50-shrink-probe-bn-batches', '100',
            '--resnet50-min-compressed-prec1', '70.0',
            '--flops-aware', '--flops-alpha', '0.25', '--strict-a-row-norm',
            '--a-row-norm-score', 'robust', '--dsf-mask-scope', 'prunable',
            '--resnet50-layerwise-prune', '--resnet50-layerwise-bound-margin', '0.08',
            '--resnet50-keep-stage1', '0.68', '--resnet50-keep-stage2', '0.40',
            '--resnet50-keep-stage3', '0.24', '--resnet50-keep-stage4', '0.20',
            '--resnet50-keep-transition', '0.58', '--resnet50-keep-last-block', '0.38',
            '--resnet50-conv2-keep-boost', '0.03', '--bn-calib-batches', '300',
            '--save-every', '5', '--print-freq', '200', '--save-dir', str(run / 'dsf')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='ImageNet root containing train/ and val/ class folders')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'outputs/resnet50_flops50')
    parser.add_argument('--device', default=None,
                        help='GPU index or UUID; default respects CUDA_VISIBLE_DEVICES')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--stage', choices=['all', 'dsf', 'prune', 'finetune'], default='all')
    parser.add_argument('--print-config', action='store_true',
                        help='print the recipe and commands without loading data or training')
    args = parser.parse_args()
    if args.workers < 0:
        parser.error('--workers must be non-negative')
    run, data_dir = args.run_dir.expanduser().resolve(), args.data_dir.expanduser().resolve()
    config = experiment_defaults()
    best = run / 'dsf/compress_model' / BEST_NAME
    commands = {
        'dsf': dsf_command(run, config, data_dir, args.workers),
        'prune': [sys.executable, '-u', str(HERE / 'prune.py'), '--run-dir', str(run),
                  '--artifact', str(best), '--data-dir', str(data_dir), '--workers', str(args.workers)],
        'finetune': [sys.executable, '-u', str(HERE / 'finetune.py'), '--run-dir', str(run),
                     '--data-dir', str(data_dir), '--workers', str(args.workers), '--seed', str(config['seed'])],
    }
    if args.print_config:
        print(json.dumps({'config': config, 'commands': commands}, indent=2))
        return

    run.mkdir(parents=True, exist_ok=True)
    with (run / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = run / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise RuntimeError('Run directory contains a different recipe; choose a new --run-dir.')
        atomic_json(config_path, config)
        env = os.environ.copy()
        if args.device is not None:
            env['CUDA_VISIBLE_DEVICES'] = args.device
        for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
            env.pop(name, None)
        env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')

        def execute(stage, command):
            print(f'Running {stage}; outputs: {run}', flush=True)
            atomic_json(run / f'{stage}_command.json', command)
            subprocess.run(command, cwd=HERE, env=env, check=True)

        stages = ['dsf', 'prune', 'finetune'] if args.stage == 'all' else [args.stage]
        for stage in stages:
            if stage == 'dsf':
                marker = run / 'dsf_completed.json'
                if marker.exists():
                    record = json.loads(marker.read_text())
                    if not best.exists() or hashlib.sha256(best.read_bytes()).hexdigest() != record['best_sha256']:
                        raise RuntimeError('Completed DSF checkpoint is missing or changed.')
                    continue
                if list((run / 'dsf/compress_model').glob('*latest.pth')):
                    raise RuntimeError('DSF was interrupted. Preserve this run and use a new --run-dir; '
                                       'DSF optimizer resume is not implemented.')
                execute(stage, commands[stage])
                final = best.with_name(BEST_NAME.replace('_best.pth', '_ft24.pth'))
                if not best.is_file() or not final.is_file():
                    raise RuntimeError('Expected outputs from all 25 DSF epochs were not found.')
                atomic_json(marker, {'epochs': 25, 'best_sha256': hashlib.sha256(best.read_bytes()).hexdigest()})
            elif stage == 'prune':
                if not (run / 'dsf_completed.json').is_file():
                    raise RuntimeError('Complete the DSF stage before structured pruning.')
                ready = run / 'initialization_status.json'
                if ready.exists() and json.loads(ready.read_text()).get('state') == 'ready':
                    continue
                execute(stage, commands[stage])
            else:
                status = run / 'training/status.json'
                if status.exists() and json.loads(status.read_text()).get('state') == 'completed':
                    print(f'Fine-tuning already completed: {status}', flush=True)
                    continue
                command = commands[stage].copy()
                latest = run / 'training/latest.pth'
                if latest.exists():
                    command.extend(['--resume', str(latest)])
                execute(stage, command)


if __name__ == '__main__':
    main()
