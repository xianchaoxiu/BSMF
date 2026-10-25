#!/usr/bin/env python3
"""Single-GPU DSFP recovery with fixed inference cost and full ImageNet validation."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from thop import profile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))
from datasets import get_imagenet
from models.hand_resnet50 import pruned_resnet50
from joint_budget import check_budget


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def clean_state(state):
    result = {}
    for key, value in state.items():
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        if key.rsplit('.', 1)[-1] not in ('total_ops', 'total_params'):
            result[key] = value
    return result


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def learning_rate(epoch_fraction, args):
    if epoch_fraction < args.warmup_epochs:
        return args.warmup_lr + (args.lr - args.warmup_lr) * epoch_fraction / args.warmup_epochs
    progress = (epoch_fraction - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
    return args.min_lr + (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(progress, 1))) / 2


def parameter_groups(model, decay):
    # Do not penalize BN affine parameters or biases.
    regular, unregular = [], []
    for parameter in model.parameters():
        (unregular if parameter.ndim <= 1 else regular).append(parameter)
    return [{'params': regular, 'weight_decay': decay}, {'params': unregular, 'weight_decay': 0.0}]


def train_transform(augmented):
    ops = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    if augmented:
        ops.append(transforms.RandAugment(num_ops=2, magnitude=5))
    ops += [transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    if augmented:
        ops.append(transforms.RandomErasing(p=0.1))
    return transforms.Compose(ops)


def loader(dataset, args, training, epoch=0):
    generator = torch.Generator().manual_seed(args.seed + epoch)
    return torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=training, drop_last=training,
        num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker,
        generator=generator, persistent_workers=False,
        **({'prefetch_factor': 2} if args.workers else {}))


@torch.inference_mode()
def validate(model, val_loader):
    """FP32, one center crop per image; no padded/repeated validation samples."""
    model.eval()
    correct1 = correct5 = count = 0
    loss_sum = 0.0
    start = time.monotonic()
    for images, target in val_loader:
        images = images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        target = target.cuda(non_blocking=True)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, target, reduction='sum').item()
        top = logits.topk(5, dim=1).indices
        correct1 += (top[:, 0] == target).sum().item()
        correct5 += (top == target[:, None]).any(dim=1).sum().item()
        count += target.numel()
    if count != 50000:
        raise RuntimeError(f'Validation must cover exactly 50000 images, got {count}')
    return {'top1': 100 * correct1 / count, 'top5': 100 * correct5 / count,
            'correct1': correct1, 'correct5': correct5, 'samples': count,
            'loss': loss_sum / count, 'seconds': time.monotonic() - start}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--warmup-lr', type=float, default=0.0025)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--clean-tail-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, choices=[256], default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--evaluate', type=Path, help='Full validation of a saved checkpoint, with resource guards.')
    args = parser.parse_args()
    if args.output is None:
        args.output = args.run_dir / 'training'
    if not (0 <= args.clean_tail_epochs < args.epochs and 0 <= args.warmup_epochs < args.epochs):
        parser.error('Warmup and clean-tail epochs must be smaller than epochs.')
    return args


def main():
    args = parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES.')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.evaluate and not args.resume and (args.output / 'latest.pth').exists():
        raise RuntimeError('Existing run found; pass --resume to preserve progress.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    # Validate in FP32; use autocast during training.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    reference = json.loads((args.run_dir / 'config.json').read_text())
    summary = json.loads((args.run_dir / 'init/config/structured_pruning_summary.json').read_text())
    if summary.get('stage') != 'direct_pruning' or summary.get('preflight'):
        raise RuntimeError('Initialization must be this experiment\'s actual direct DSF pruning output.')
    for name in ('epochs', 'lr', 'min_lr', 'warmup_lr', 'warmup_epochs', 'momentum',
                 'weight_decay', 'label_smoothing', 'clean_tail_epochs'):
        if getattr(args, name) != reference['finetune'][name]:
            raise RuntimeError(f'Fine-tuning recipe mismatch: {name}')
    cfg = summary['cfg']
    model = pruned_resnet50(cfg=copy.deepcopy(cfg), dataset='imagenet').eval()
    initial_path = args.run_dir / 'init/finetune_model/resnet50_structured_pruned_init_bncalibrated.pth'
    checkpoint_path = args.evaluate or args.resume or initial_path
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(clean_state(checkpoint['model_state_dict']), strict=True)
    probe = copy.deepcopy(model)
    with torch.inference_mode():
        flops, _ = profile(probe, inputs=(torch.zeros(1, 3, 224, 224),), verbose=False)
    del probe
    params = sum(p.numel() for p in model.parameters())
    if flops > reference['max_flops'] or params > reference['max_params']:
        raise RuntimeError(f'Resource limit exceeded: FLOPs={flops}, params={params}')
    fdrop, pdrop = check_budget(flops, params, reference)
    model = model.cuda().to(memory_format=torch.channels_last)
    resources = {'flops': int(flops), 'parameters': params, 'flops_drop_percent': fdrop, 'params_drop_percent': pdrop,
                 'max_flops': reference['max_flops'], 'max_parameters': reference['max_params'],
                 'input_shape': [1, 3, 224, 224], 'cfg': cfg,
                 'gpu': torch.cuda.get_device_name(0), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'batch_size': 256, 'ema': False, 'kd': False, 'swa': False,
                 'ensemble': False, 'test_time_augmentation': False,
                 'torch': torch.__version__, 'validation_precision': 'float32'}
    atomic_json(args.output / 'resources.json', resources)
    print('RESOURCE_GUARD_PASS', json.dumps(resources), flush=True)

    classes, train_data, val_data = get_imagenet(Path(args.data_dir))
    if classes != 1000 or len(train_data) != 1281167 or len(val_data) != 50000:
        raise RuntimeError('Unexpected ImageNet dataset size or class count.')
    if train_data.class_to_idx != val_data.class_to_idx:
        raise RuntimeError('Train and validation class mappings differ.')
    val_loader = loader(val_data, args, False)
    if args.evaluate:
        metrics = validate(model, val_loader)
        result = {'checkpoint': str(checkpoint_path), **metrics, **resources}
        atomic_json(args.output / 'verification.json', result)
        print('VERIFIED', json.dumps(result), flush=True)
        return

    optimizer = torch.optim.SGD(parameter_groups(model, args.weight_decay),
                                lr=args.lr, momentum=args.momentum, nesterov=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_top1, best_epoch, start_epoch = -1.0, -1, 0
    if args.resume:
        # Reject an accidental recipe change when restoring optimizer/RNG state.
        saved_args = checkpoint['recipe']
        for name in ('epochs', 'lr', 'min_lr', 'warmup_lr', 'warmup_epochs', 'momentum',
                     'weight_decay', 'label_smoothing', 'clean_tail_epochs', 'batch_size', 'seed'):
            if getattr(args, name) != saved_args[name]:
                raise ValueError(f'Resume recipe mismatch: {name}')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        best_top1, best_epoch = checkpoint['best_top1'], checkpoint['best_epoch']
        start_epoch = checkpoint['epoch'] + 1
        random.setstate(checkpoint['rng_python'])
        np.random.set_state(checkpoint['rng_numpy'])
        torch.set_rng_state(checkpoint['rng_torch'])
        torch.cuda.set_rng_state(checkpoint['rng_cuda'])
    recipe = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    recipe.update(training_precision=str(dtype), augmentation=f'RandAugment(2,5), RandomErasing(0.1); disabled for last {args.clean_tail_epochs} epochs',
                  weight_decay_on_norm_and_bias=0.0, nesterov=True, initial_checkpoint=str(initial_path),
                  initial_checkpoint_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest())
    atomic_json(args.output / 'recipe.json', recipe)
    print('RECIPE', json.dumps(recipe), flush=True)

    def save_checkpoint(path, epoch, metrics):
        atomic_save(path, {'epoch': epoch, 'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                          'prec1': metrics['top1'], 'prec5': metrics['top5'],
                          'best_top1': best_top1, 'best_epoch': best_epoch, 'metrics': metrics,
                          'cfg': cfg, 'resources': resources, 'recipe': recipe,
                          'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                          'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state()})

    if args.resume:
        prior = json.loads((args.output / 'status.json').read_text()) if (args.output / 'status.json').exists() else {}
        prior.update(state='training', completed_epochs=start_epoch, resumed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        atomic_json(args.output / 'status.json', prior)

    if not args.resume:
        initial_metrics = validate(model, val_loader)
        best_top1 = initial_metrics['top1']
        save_checkpoint(args.output / 'best.pth', -1, initial_metrics)
        atomic_json(args.output / 'status.json', {'state': 'training', 'completed_epochs': 0, 'best_top1': best_top1, 'best_epoch': -1, 'initial_metrics': initial_metrics})
        print('INITIAL_VALIDATION', json.dumps(initial_metrics), flush=True)

    for epoch in range(start_epoch, args.epochs):
        train_data.transform = train_transform(epoch < args.epochs - args.clean_tail_epochs)
        train_loader = loader(train_data, args, True, epoch)
        model.train()
        start = time.monotonic()
        loss_sum = count = correct = 0
        torch.cuda.reset_peak_memory_stats()
        for step, (images, target) in enumerate(train_loader):
            if images.shape[0] != 256:
                raise RuntimeError('Every training step must use batch size 256.')
            lr = learning_rate(epoch + step / len(train_loader), args)
            for group in optimizer.param_groups:
                group['lr'] = lr
            images = images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=dtype):
                logits = model(images)
                loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch={epoch}, step={step}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Fail on nonfinite gradients; clip only exceptional gradient spikes.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * 256
            count += 256
            correct += (logits.argmax(1) == target).sum().item()
            if step % 100 == 0:
                progress = {'epoch': epoch + 1, 'epochs': args.epochs, 'step': step + 1,
                            'steps': len(train_loader), 'lr': lr, 'loss': loss_sum / count,
                            'train_top1': 100 * correct / count, 'grad_norm': float(grad_norm),
                            'images_per_second': count / (time.monotonic() - start),
                            'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30}
                print('TRAIN', json.dumps(progress), flush=True)
                atomic_json(args.output / 'progress.json', progress)
        train_seconds = time.monotonic() - start
        metrics = validate(model, val_loader)
        improved = metrics['top1'] > best_top1
        if improved:
            best_top1, best_epoch = metrics['top1'], epoch
        row = {'epoch': epoch + 1, 'lr': lr, 'train_loss': loss_sum / count, 'train_top1': 100 * correct / count, 'train_samples': count, 'train_seconds': train_seconds, 'validation': metrics, 'best_top1': best_top1, 'best_epoch': best_epoch + 1}
        # Store only best + latest; retain source and initial checkpoint separately.
        save_checkpoint(args.output / 'latest.pth', epoch, metrics)
        if improved:
            save_checkpoint(args.output / 'best.pth', epoch, metrics)
        with (args.output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        atomic_json(args.output / 'status.json', {'state': 'training', 'completed_epochs': epoch + 1, **row})
        print('EPOCH_RESULT', json.dumps(row), flush=True)

    final_checkpoint = torch.load(args.output / 'best.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(final_checkpoint['model_state_dict'], strict=True)
    verification = validate(model, val_loader)
    status = {'state': 'completed', 'completed_epochs': args.epochs, 'best_top1': best_top1, 'best_epoch': best_epoch + 1, 'verification': verification, 'resources': resources, 'best_checkpoint': str(args.output / 'best.pth')}
    atomic_json(args.output / 'status.json', status)
    print('COMPLETED', json.dumps(status), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Never label an aborted experiment as completed or successful.
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        raise
