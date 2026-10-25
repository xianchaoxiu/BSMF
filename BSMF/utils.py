import json
import os
import sys
import time
from datetime import datetime

import torch
import math
import torch.distributed as dist


def should_save_epoch_checkpoint(epoch, total_epochs, save_every):
    """Keep restart milestones without writing a checkpoint every epoch."""
    epoch = int(epoch)
    total_epochs = int(total_epochs)
    save_every = int(save_every)
    return (
        epoch == 0
        or epoch + 1 == total_epochs
        or (save_every > 0 and (epoch + 1) % save_every == 0)
    )

def is_cifar_resnet(args):
    """Identify CIFAR ResNet-56 and ResNet-110 models."""
    return (str(getattr(args, 'model', '')).lower() in ('resnet56', 'resnet110')
            and str(getattr(args, 'dataset', '')).lower() in ('cifar10', 'cifar100'))


def is_vgg16_cifar(args):
    return (str(getattr(args, 'model', '')).lower() == 'vgg16'
            and str(getattr(args, 'dataset', '')).lower() in ('cifar10', 'cifar100'))


def admm_cosine_lr(base_lr, total_epochs, epoch, step, len_iter, warm_epochs=5, min_lr=0.0):
    """Compute cosine decay with per-batch linear warmup and a learning-rate floor."""
    base_lr = float(base_lr)
    total_epochs = int(total_epochs)
    epoch = int(epoch)
    step = int(step)
    len_iter = max(1, int(len_iter))
    warm_epochs = max(0, int(warm_epochs))
    min_lr = float(min_lr)

    if total_epochs <= warm_epochs:
        lr = base_lr
    else:
        progress = (epoch - warm_epochs) / float(total_epochs - warm_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    if warm_epochs > 0 and epoch < warm_epochs:
        lr *= float(1 + step + epoch * len_iter) / float(warm_epochs * len_iter)
    return lr

def adjust_learning_rate(optimizer, epoch, step, len_iter, args, base_lr=None, total_epochs=None, lr_type=None):
    base_lr = float(base_lr if base_lr is not None else getattr(args, 'current_lr', getattr(args, 'lr', 0.1)))
    total_epochs = int(total_epochs if total_epochs is not None else getattr(args, 'current_total_epochs', getattr(args, 'fine_epochs', getattr(args, 'epochs', 0))))
    lr_type = lr_type if lr_type is not None else getattr(args, 'current_lr_type', getattr(args, 'lr_type', 'cos'))
    warm_epochs = int(getattr(args, 'current_warm_epochs', 5))
    min_lr = float(getattr(args, 'current_min_lr', 0.0))

    if lr_type == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor += 1
        lr = base_lr * (0.1 ** factor)
    elif lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor += 1
        lr = base_lr * (0.5 ** factor)
    elif lr_type == 'cos' and is_vgg16_cifar(args):
        # VGG16 cosine decay with per-batch warmup.
        lr = admm_cosine_lr(base_lr, total_epochs, epoch, step, len_iter,
                            warm_epochs=warm_epochs, min_lr=min_lr)
    elif lr_type == 'cos' and is_cifar_resnet(args) and warm_epochs == 5 and min_lr == 0.0:
        # Keep multiplication before division to preserve floating-point rounding.
        if total_epochs <= 5:
            lr = base_lr
        else:
            lr = 0.5 * base_lr * (1 + math.cos(math.pi * (epoch - 5) / (total_epochs - 5)))
    elif lr_type == 'cos':
        if total_epochs <= warm_epochs:
            lr = base_lr
        else:
            progress = (epoch - warm_epochs) / max(1, total_epochs - warm_epochs)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    elif lr_type == 'exp':
        decay = 0.96
        lr = base_lr * (decay ** epoch)
    elif lr_type == 'fixed':
        lr = base_lr
    else:
        raise NotImplementedError(f'Unknown lr_type: {lr_type}')

    if (lr_type != 'fixed' and not (lr_type == 'cos' and is_vgg16_cifar(args))
            and warm_epochs > 0 and epoch < warm_epochs):
        lr = lr * float(1 + step + epoch * len_iter) / (float(warm_epochs) * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step == 0 and is_main_process(args):
        print('learning_rate : ', lr)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process(args=None):
    if args is not None and hasattr(args, "rank"):
        return int(getattr(args, "rank", 0)) == 0
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _to_jsonable(value.detach().cpu().item())
        return _to_jsonable(value.detach().cpu().tolist())
    return str(value)


def save_run_config(args, config_dir=None, prefix="run_args"):
    if args is None or not is_main_process(args):
        return None

    save_dir = getattr(args, "save_dir", "save_temp")
    config_dir = config_dir or os.path.join(save_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    args_dict = {key: _to_jsonable(value) for key, value in sorted(vars(args).items())}
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "args": args_dict,
    }

    json_path = os.path.join(config_dir, f"{prefix}.json")
    txt_path = os.path.join(config_dir, f"{prefix}.txt")
    cmd_path = os.path.join(config_dir, "command.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")

    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(payload["command"] + "\n")

    print(f"===> Saved run config to: {json_path}")
    return json_path


def validate(val_loader, model, criterion , args, device):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 , prec5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item() , input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and is_main_process(args):
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1 , top5 = top5))

    if is_dist_avail_and_initialized():
        stats = torch.tensor(
            [losses.sum, top1.sum, top5.sum, losses.count],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total = max(float(stats[3].item()), 1.0)
        losses.avg = float(stats[0].item() / total)
        top1.avg = float(stats[1].item() / total)
        top5.avg = float(stats[2].item() / total)

    if is_main_process(args):
        print(' * Prec@1 {top1.avg:.3f} , Prec@5 {top5.avg:.3f}'
              .format(top1=top1 , top5 = top5))

    return top1.avg , top5.avg
