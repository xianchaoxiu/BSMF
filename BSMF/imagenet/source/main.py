import argparse
import hashlib
import os
import time
import random
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.models as tv_models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import ResNet50_Weights

from utils import *
from compute_flops import *
from datasets import *
from channel_pruning_resnet import *
from compress_model import compress_model
from compress_model import _bn_calibrate
from models.built_model import *

PROJECT_ROOT = Path(__file__).resolve().parent

# Verify the official ResNet-50 V1 checkpoint checksum.
OFFICIAL_RESNET50_V1_SHA256 = (
    "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
)
OFFICIAL_RESNET50_V1_WEIGHTS = "torchvision.models.ResNet50_Weights.IMAGENET1K_V1"


def _assert_official_resnet50_v1_cache() -> None:
    weights_path = Path(torch.hub.get_dir()) / "checkpoints/resnet50-0676ba61.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(
            "Official ResNet-50 V1 weights are not cached at "
            f"{weights_path}; refusing to substitute a self-trained baseline."
        )
    digest = hashlib.sha256()
    with weights_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != OFFICIAL_RESNET50_V1_SHA256:
        raise RuntimeError(
            f"Official ResNet-50 V1 weight hash mismatch: {actual} != "
            f"{OFFICIAL_RESNET50_V1_SHA256}"
        )


def set_args():
    parser = argparse.ArgumentParser(description='Proper ResNets for CIFAR in pytorch')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--fine-epochs', default=30, type=int, metavar='N',
                        help='number of total epochs for structured-pruning finetune')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--re-lr', '--relearning-rate', default=0.01, type=float,
                        metavar='LR', help='learning rate for pruning finetune')
    parser.add_argument('--sp-momentum', dest='sp_momentum', default=None, type=float,
                        help='momentum for structured-pruning finetune; default follows --momentum')
    parser.add_argument('--sp-weight-decay', dest='sp_weight_decay', default=None, type=float,
                        help='weight decay for structured-pruning finetune; default follows --weight-decay')
    parser.add_argument('--dsf-lr', '--dsflearning-rate', dest='dsf_lr', default=0.0005, type=float,
                        metavar='LR', help='learning rate for DSF finetune')
    parser.add_argument('--dsf-warm-epochs', dest='dsf_warm_epochs', default=0, type=int,
                        help='ResNet50/ImageNet DSF warmup epochs; starts at --dsf-warm-lr')
    parser.add_argument('--dsf-warm-lr', dest='dsf_warm_lr', default=0.0, type=float,
                        help='starting LR for the DSF warmup (linear to --dsf-lr)')
    parser.add_argument('--dsf-momentum', dest='dsf_momentum', default=None, type=float,
                        help='momentum for DSF recovery finetune; default follows --momentum')
    parser.add_argument('--dsf-weight-decay', dest='dsf_weight_decay', default=None, type=float,
                        help='weight decay for DSF recovery finetune; default follows --weight-decay')
    parser.add_argument('--dsf-label-smoothing', dest='dsf_label_smoothing', default=0.05, type=float,
                        help='label smoothing used during DSF recovery finetune')
    parser.add_argument('--baseline-label-smoothing', dest='baseline_label_smoothing', default=0.0, type=float,
                        help='label smoothing used only during baseline training')
    parser.add_argument('--sp-label-smoothing', dest='sp_label_smoothing', default=0.1, type=float,
                        help='label smoothing used during structured-pruning finetune')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--device', default='0', type=str, metavar='M',
                        help='GPU id(s), e.g. 0 or 0,1,2; multi-GPU is enabled only for ResNet50/ImageNet')
    parser.add_argument('--ddp-timeout-minutes', dest='ddp_timeout_minutes', default=120, type=int,
                        help='DDP/NCCL collective timeout in minutes for ResNet50/ImageNet torchrun jobs')
    parser.add_argument('--model', default='resnet56',
                        choices=['resnet18', 'resnet20', 'resnet50', 'vgg19', 'vgg16', 'resnet56', 'resnet101', 'resnet110'],
                        help='model')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--data-dir', default=str(PROJECT_ROOT / 'data'), type=str,
                        help='dataset root directory')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model when supported')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit)')
    parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str,
                        help='directory used to save the trained models')
    parser.add_argument('--save-every', dest='save_every', type=int, default=10,
                        help='save numbered restart checkpoints every N epochs; best/latest are always kept')
    parser.add_argument('--save-eval-baseline', action='store_true',
                        help='save the evaluated model as the baseline checkpoint, useful for pretrained ImageNet baselines')
    parser.add_argument('--lamb', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=random.randint(1, 1000))
    parser.add_argument('--data', type=float, default=1.)
    parser.add_argument('--ratio', type=float, default=0.47)
    parser.add_argument('--p', type=float, default=0.5, help='p < 1 norm')
    parser.add_argument('--q', type=float, default=0.5, help='q < 1 norm')
    parser.add_argument('--dsf-sparsity', dest='dsf_sparsity', type=float, default=0.2,
                        help='target sparsity for doublesparse factorization')
    parser.add_argument('--shrink-iters', dest='shrink_iters', type=int, default=10)
    parser.add_argument('--group-small-thr', dest='group_small_thr', type=float, default=5e-2,
                        help='threshold used when reporting small A-row/B-col groups during DSF shrink')
    parser.add_argument('--resnet50-shrink-target-a-row-small', type=float, default=None,
                        help='ResNet50/ImageNet only: restore/stop DSF shrink near this mean A_row_small ratio')
    parser.add_argument('--resnet50-shrink-target-tolerance', type=float, default=0.05,
                        help='ResNet50/ImageNet only: tolerance around --resnet50-shrink-target-a-row-small')
    parser.add_argument('--resnet50-shrink-min-iters', type=int, default=1,
                        help='ResNet50/ImageNet only: minimum DSF shrink iterations before target early-stop')
    parser.add_argument('--resnet50-shrink-restore-best', action='store_true',
                        help='ResNet50/ImageNet only: restore the A/B snapshot whose A_row_small is closest to target')
    parser.add_argument('--resnet50-shrink-select-by-val', action='store_true',
                        help='ResNet50/ImageNet only: validate each shrink snapshot and keep the best one')
    parser.add_argument('--resnet50-shrink-probe-bn-batches', type=int, default=80,
                        help='ResNet50/ImageNet only: BN calibration batches before each shrink snapshot validation')
    parser.add_argument('--resnet50-min-compressed-prec1', type=float, default=None,
                        help='ResNet50/ImageNet only: abort if compressed pre-finetune top1 is below this value')
    parser.add_argument('--resnet50-min-sparse-init-prec1', type=float, default=None,
                        help='ResNet50/ImageNet only: abort if calibrated structured-pruned init is below this value')
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--dsf', action='store_true',
                        help='whether to apply DSF factorization after base training')
    parser.add_argument('--adjust', type=float, default=0.3,
                        help='sensitive-layer minimum keep ratio for structured pruning')
    parser.add_argument('--compressed-epoch', dest='compressed_epoch', type=int, default=3,
                        help='number of finetune epochs inside compress_model')
    parser.add_argument('--num_re_epochs', type=int, default=3, metavar='R')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='E')
    parser.add_argument('--sp', '--structured-pruning', action='store_true', default=False,
                        help='structured pruning')
    parser.add_argument('--load-dsf-artifacts', dest='load_dsf_artifacts', default='', type=str,
                        help='load DSF channel_scores/weff masks from a checkpoint before structured pruning')
    parser.add_argument('--load-dsf-model-state', dest='load_dsf_model_state', action='store_true',
                        help='when --load-dsf-artifacts points to a DSF checkpoint, also load its model_state_dict before structured pruning')
    parser.add_argument('--up', '--unstructured-pruning', action='store_true', default=False,
                        help='unstructured pruning')
    parser.add_argument('--lr-type', dest='lr_type', type=str, default='cos',
                        choices=['cos', 'step', 'step_5', 'exp', 'fixed'],
                        help='base-training lr schedule')
    parser.add_argument('--re-lr-type', dest='re_lr_type', type=str, default='cos',
                        choices=['cos', 'step', 'step_5', 'exp', 'fixed'],
                        help='structured-pruning finetune lr schedule')
    parser.add_argument('--lambda_A', type=float, default=0.003)
    parser.add_argument('--lambda_B', type=float, default=0.0002)
    parser.add_argument('--rho', type=float, default=0.01)
    parser.add_argument('--warm-epochs', dest='warm_epochs', type=int, default=0,
                        help='optional warmup epochs, mainly for pretrained imagenet-style models')
    parser.add_argument('--warm-lr', dest='warm_lr', type=float, default=0.01,
                        help='warmup lr when warmup is enabled')
    parser.add_argument('--sp-warm-epochs', dest='sp_warm_epochs', type=int, default=None,
                        help='structured-pruning finetune warmup epochs (default: 5)')
    parser.add_argument('--sp-min-lr', dest='sp_min_lr', type=float, default=0.0,
                        help='minimum lr floor for structured-pruning finetune cosine schedule')
    parser.add_argument('--bn-calib-batches', dest='bn_calib_batches', type=int, default=200)
    parser.add_argument('--dsf-rank-mode', type=str, default='auto',
                        choices=['auto', 'fixed'],
                        help='auto selects per-layer rank by spectral energy and parameter budget; fixed uses --dsf-rank-ratio')
    parser.add_argument('--dsf-rank-ratio', type=float, default=0.72,
                        help='fixed rank ratio, used when --dsf-rank-mode fixed')
    parser.add_argument('--dsf-rank-energy', type=float, default=0.90,
                        help='target singular-value energy for automatic DSF rank selection')
    parser.add_argument('--dsf-rank-param-frac', type=float, default=0.90,
                        help='cap A/B factorized params to this fraction of original layer params; set >=1 to disable')
    parser.add_argument('--dsf-min-rank', type=int, default=1,
                        help='minimum DSF factorization rank after low-rank parameter cap')
    parser.add_argument('--dsf-init', type=str, default='svd',
                        choices=['svd', 'factorize_svd'])
    parser.add_argument('--flops-aware', action='store_true',
                    help='use FLOPs-aware channel ranking for structured pruning')
    parser.add_argument('--flops-alpha', type=float, default=0.25,
                    help='strength of FLOPs-aware pruning score')
    parser.add_argument('--strict-a-row-norm', dest='strict_a_row_norm', action='store_true',
                    help='use DSF A-row L2 norms as the only structured-pruning channel score')
    parser.add_argument('--no-strict-a-row-norm', dest='strict_a_row_norm', action='store_false',
                    help='allow fallback scores such as convolution weight magnitude')
    parser.set_defaults(strict_a_row_norm=True)
    parser.add_argument('--a-row-norm-score', type=str, default='robust',
                    choices=['robust', 'raw', 'aib_mix', 'aib_tie', 'random'],
                    help='channel score derived from DSF A-row norms; robust keeps the previous cross-layer normalization; aib_mix blends A-row norms with ||A_i B||_2 contribution scores; aib_tie keeps A-row dominant and uses ||A_i B||_2 only as a small tie-break correction')
    parser.add_argument('--a-row-aib-alpha', type=float, default=0.85,
                    help='weight for A-row norm when a-row-norm-score=aib_mix; the remaining weight goes to ||A_i B||_2')
    parser.add_argument('--a-row-aib-tie-beta', type=float, default=0.02,
                    help='small tie-break weight for ||A_i B||_2 when a-row-norm-score=aib_tie')
    parser.add_argument('--target-flops-drop', type=float, default=None,
                    help='target FLOPs drop for final structured pruning; use 40 or 0.40 for 40%%')
    parser.add_argument('--target-flops-tolerance', type=float, default=0.05,
                    help='allowed absolute FLOPs-drop error in percentage points')
    parser.add_argument('--dsf-mask-scope', type=str, default='prunable',
                        choices=['prunable', 'all'],
                        help='layers used to build DSF A-row-norm hard masks')
    parser.add_argument('--no-dsf-hard-mask', dest='use_dsf_hard_mask', action='store_false',
                        help='ignore DSF hard masks during structured pruning')
    parser.set_defaults(use_dsf_hard_mask=True)
    parser.add_argument('--min-keep-normal', type=float, default=0.15,
                    help='minimum keep ratio for normal prunable layers')
    parser.add_argument('--min-keep-medium', type=float, default=0.35,
                    help='minimum keep ratio for medium-sensitive layers')
    parser.add_argument('--min-keep-sensitive', type=float, default=0.50,
                    help='minimum keep ratio for sensitive layers')
    parser.add_argument('--max-keep-normal', type=float, default=1.0,
                    help='maximum keep ratio for normal prunable layers')
    parser.add_argument('--max-keep-medium', type=float, default=1.0,
                    help='maximum keep ratio for medium-sensitive layers')
    parser.add_argument('--max-keep-sensitive', type=float, default=1.0,
                    help='maximum keep ratio for sensitive layers')
    parser.add_argument('--resnet50-layerwise-prune', action='store_true',
                    help='use ResNet-50/ImageNet layer-wise pruning budgets instead of one global channel threshold')
    parser.add_argument('--resnet50-keep-stage1', type=float, default=0.75,
                    help='keep ratio for ResNet-50 layer1 conv1/conv2 when layer-wise pruning is enabled')
    parser.add_argument('--resnet50-keep-stage2', type=float, default=0.58,
                    help='keep ratio for ResNet-50 layer2 conv1/conv2 when layer-wise pruning is enabled')
    parser.add_argument('--resnet50-keep-stage3', type=float, default=0.42,
                    help='keep ratio for ResNet-50 layer3 conv1/conv2 when layer-wise pruning is enabled')
    parser.add_argument('--resnet50-keep-stage4', type=float, default=0.50,
                    help='keep ratio for ResNet-50 layer4 conv1/conv2 when layer-wise pruning is enabled')
    parser.add_argument('--resnet50-keep-transition', type=float, default=0.70,
                    help='keep ratio for ResNet-50 transition/downsampling bottleneck conv1/conv2 when layer-wise pruning is enabled')
    parser.add_argument('--resnet50-keep-last-block', type=float, default=0.70,
                    help='keep ratio for ResNet-50 last bottleneck conv1/conv2 when layer-wise pruning is enabled')
    parser.add_argument('--resnet50-conv2-keep-boost', type=float, default=0.10,
                    help='extra keep ratio for ResNet-50 bottleneck conv2 when layer-wise pruning is enabled')
    parser.add_argument('--resnet50-layerwise-bound-margin', type=float, default=0.0,
                    help='ResNet-50/ImageNet only: allow final A-row pruning to move within keep_ratio +/- margin')
    parser.add_argument('--resnet56-layerwise-prune', action='store_true',
                    help='use ResNet-56/CIFAR layer-wise protection bounds during final structured pruning')
    parser.add_argument('--resnet56-layerwise-bounds-only', action='store_true',
                    help='alias for the ResNet-56 bounds-based DSF-mask pruning policy')
    parser.add_argument('--resnet56-layerwise-bound-margin', type=float, default=0.12,
                    help='margin around each ResNet-56 keep ratio; A-row-norm masks choose the final keep count inside the bounds')
    parser.add_argument('--resnet56-adaptive-protect', action='store_true',
                    help='for ResNet-56, protect sensitive blocks with A-row-norm adaptive floors instead of fixed stage bounds')
    parser.add_argument('--resnet56-adaptive-span', type=float, default=0.10,
                    help='range used to vary ResNet-56 adaptive protection floors by per-layer A-row-norm strength')
    parser.add_argument('--resnet56-adaptive-max-keep', type=float, default=1.0,
                    help='optional upper keep ratio for ResNet-56 adaptive protection; default disables max clipping')
    parser.add_argument('--resnet56-adaptive-min-stage2', type=float, default=None,
                    help='optional minimum keep ratio for ordinary ResNet-56 stage2 blocks under adaptive protection')
    parser.add_argument('--resnet56-adaptive-min-stage3', type=float, default=None,
                    help='optional minimum keep ratio for ordinary ResNet-56 stage3 blocks under adaptive protection')
    parser.add_argument('--resnet56-adaptive-max-stage2', type=float, default=None,
                    help='optional maximum keep ratio for ordinary ResNet-56 stage2 blocks under adaptive protection')
    parser.add_argument('--resnet56-adaptive-max-stage3', type=float, default=None,
                    help='optional maximum keep ratio for ordinary ResNet-56 stage3 blocks under adaptive protection')
    parser.add_argument('--resnet56-adaptive-max-last-block', type=float, default=None,
                    help='optional maximum keep ratio for final ResNet-56 blocks under adaptive protection')
    parser.add_argument('--resnet56-keep-stage1', type=float, default=0.70,
                    help='keep ratio for ResNet-56 layer1 conv_bn1 when layer-wise pruning is enabled')
    parser.add_argument('--resnet56-keep-stage2', type=float, default=0.40,
                    help='keep ratio for ResNet-56 ordinary layer2 conv_bn1 when layer-wise pruning is enabled')
    parser.add_argument('--resnet56-keep-stage3', type=float, default=0.30,
                    help='keep ratio for ResNet-56 ordinary layer3 conv_bn1 when layer-wise pruning is enabled')
    parser.add_argument('--resnet56-stage3-tail-low-keep', type=float, default=None,
                    help='optional lower keep ratio for the last ordinary ResNet-56 stage3 blocks')
    parser.add_argument('--resnet56-stage3-tail-low-count', type=int, default=0,
                    help='number of ordinary ResNet-56 stage3 blocks before the protected tail that use --resnet56-stage3-tail-low-keep')
    parser.add_argument('--resnet56-keep-transition', type=float, default=0.65,
                    help='keep ratio for ResNet-56 layer2/layer3 block_0 conv_bn1 when layer-wise pruning is enabled')
    parser.add_argument('--resnet56-keep-last-block', type=float, default=0.48,
                    help='minimum keep ratio for the final ResNet-56 blocks when layer-wise pruning is enabled')
    parser.add_argument('--resnet56-last-block-count', type=int, default=2,
                    help='number of final blocks protected by --resnet56-keep-last-block')
    parser.add_argument('--resnet110-layerwise-prune', action='store_true',
                    help='use ResNet-110/CIFAR layer-wise keep budgets during final structured pruning')
    parser.add_argument('--resnet110-layerwise-bounds-only', action='store_true',
                    help='for ResNet-110, use layer-wise keep budgets as min/max bounds around the DSF hard mask instead of exact keep counts')
    parser.add_argument('--resnet110-layerwise-bound-margin', type=float, default=0.12,
                    help='margin around each ResNet-110 layer-wise keep ratio when --resnet110-layerwise-bounds-only is enabled')
    parser.add_argument('--resnet110-keep-stage1', type=float, default=0.52,
                    help='keep ratio for ordinary ResNet-110 layer1 conv_bn1 when layer-wise pruning is enabled')
    parser.add_argument('--resnet110-keep-stage2', type=float, default=0.42,
                    help='keep ratio for ordinary ResNet-110 layer2 conv_bn1 when layer-wise pruning is enabled')
    parser.add_argument('--resnet110-keep-stage3', type=float, default=0.40,
                    help='keep ratio for ordinary ResNet-110 layer3 conv_bn1 when layer-wise pruning is enabled')
    parser.add_argument('--resnet110-keep-transition', type=float, default=0.78,
                    help='keep ratio for ResNet-110 layer2/layer3 block_0 conv_bn1 when layer-wise pruning is enabled')
    parser.add_argument('--resnet110-keep-early-block', type=float, default=0.62,
                    help='minimum keep ratio for early ResNet-110 blocks when layer-wise pruning is enabled')
    parser.add_argument('--resnet110-early-block-count', type=int, default=2,
                    help='number of early blocks protected by --resnet110-keep-early-block')
    parser.add_argument('--resnet110-keep-last-block', type=float, default=0.52,
                    help='minimum keep ratio for final ResNet-110 blocks when layer-wise pruning is enabled')
    parser.add_argument('--resnet110-last-block-count', type=int, default=2,
                    help='number of final blocks protected by --resnet110-keep-last-block')
    parser.add_argument('--lambda-B-sensitive-scale', type=float, default=0.50,
                    help='lambda_B multiplier for sensitive layers in DSF shrink')
    parser.add_argument('--lambda-B-medium-scale', type=float, default=0.75,
                    help='lambda_B multiplier for medium-sensitive layers in DSF shrink')
    return parser.parse_args()


def fix(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args, num_classes, device):
    model_name = args.model.lower()

    if model_name == 'resnet56':
        model = resnet56(num_classes=num_classes)
    elif model_name == 'resnet110':
        model = resnet110(num_classes=num_classes)
    elif model_name == 'resnet20':
        model = resnet20(num_classes=num_classes)
    elif model_name == 'resnet50':
        if str(getattr(args, 'dataset', '')).lower() == 'imagenet' and args.pretrained:
            _assert_official_resnet50_v1_cache()
        weights = ResNet50_Weights.IMAGENET1K_V1 if args.pretrained else None
        model = tv_models.resnet50(weights=weights)
        if model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f'Unsupported model: {args.model}')

    return model.to(device)


def is_resnet50_imagenet(args):
    return (
        str(getattr(args, 'model', '')).lower() == 'resnet50'
        and str(getattr(args, 'dataset', '')).lower() == 'imagenet'
    )


def setup_for_distributed(is_master):
    import builtins

    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def init_distributed_mode(args):
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0
    args.global_batch_size = int(args.batch_size)

    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        setup_for_distributed(True)
        return

    if not is_resnet50_imagenet(args):
        raise RuntimeError('torchrun/DDP is only supported for ResNet50/ImageNet in this script.')
    if not torch.cuda.is_available():
        raise RuntimeError('DDP requires CUDA for this script.')

    args.distributed = True
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=timedelta(minutes=int(getattr(args, 'ddp_timeout_minutes', 120))),
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)

    if args.global_batch_size % args.world_size != 0:
        raise ValueError(
            f'--batch-size is treated as global batch size under DDP and must be '
            f'divisible by world_size={args.world_size}; got {args.global_batch_size}'
        )
    args.batch_size = args.global_batch_size // args.world_size
    print(
        f"===> Enable DDP for ResNet50/ImageNet: world_size={args.world_size}, "
        f"global_batch={args.global_batch_size}, per_gpu_batch={args.batch_size}",
    )


def cleanup_distributed(args):
    if getattr(args, 'distributed', False) and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process_local(args):
    return int(getattr(args, 'rank', 0)) == 0


def unwrap_model(model):
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def checkpoint_state_dict(model):
    return unwrap_model(model).state_dict()


def load_state_dict_into_model(model, state_dict, strict=False):
    return unwrap_model(model).load_state_dict(state_dict, strict=strict)


def strip_module_prefix(state_dict):
    stripped = {}
    for key, value in state_dict.items():
        while key.startswith('module.'):
            key = key[len('module.'):]
        stripped[key] = value
    return stripped


def parse_cuda_device_ids(device_arg):
    text = str(device_arg).strip().lower()
    if text in ('', 'cpu', '-1', 'none'):
        return []
    if text == 'all':
        return list(range(torch.cuda.device_count()))

    device_ids = []
    for part in text.split(','):
        part = part.strip()
        if part.startswith('cuda:'):
            part = part.split(':', 1)[1]
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"Unsupported --device value: {device_arg}")
        device_ids.append(int(part))
    if len(device_ids) == 0:
        raise ValueError(f"Unsupported --device value: {device_arg}")
    return device_ids


def resolve_device_and_parallel_ids(args):
    if getattr(args, 'distributed', False):
        device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(device)
        return device, []

    if str(args.device).strip().lower() in ('cpu', '-1') or not torch.cuda.is_available():
        return torch.device('cpu'), []

    device_ids = parse_cuda_device_ids(args.device)
    if len(device_ids) == 0:
        return torch.device('cpu'), []

    available = torch.cuda.device_count()
    for gpu_id in device_ids:
        if gpu_id < 0 or gpu_id >= available:
            raise ValueError(
                f"GPU id {gpu_id} is not available. "
                f"Visible CUDA device count: {available}"
            )

    if len(device_ids) > 1 and not is_resnet50_imagenet(args):
        print(
            "[WARN] Multi-GPU is only enabled for ResNet50/ImageNet in this script; "
            f"using cuda:{device_ids[0]} for {args.model}/{args.dataset}."
        )
        device_ids = device_ids[:1]

    primary_gpu = device_ids[0]
    torch.cuda.set_device(primary_gpu)
    parallel_ids = device_ids if len(device_ids) > 1 and is_resnet50_imagenet(args) else []
    return torch.device(f'cuda:{primary_gpu}'), parallel_ids


def maybe_wrap_resnet50_imagenet_parallel(model, args, device_ids):
    if getattr(args, 'distributed', False):
        if not is_resnet50_imagenet(args):
            return model
        print(
            f"===> Wrap model with DistributedDataParallel on local_rank={args.local_rank}",
        )
        return DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )

    if len(device_ids) <= 1:
        return model
    if not is_resnet50_imagenet(args):
        return model
    print(f"===> Enable DataParallel for ResNet50/ImageNet on GPUs: {device_ids}")
    return nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])


def _filter_incompatible_state_dict(model, state_dict):
    model_state = unwrap_model(model).state_dict()
    state_dict = strip_module_prefix(state_dict)
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, tuple(value.shape), None))
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = value
    return filtered, skipped


def maybe_load_resume(model, args, device):
    best_prec1 = 0.0
    if is_resnet50_imagenet(args) and args.pretrained and args.resume:
        raise RuntimeError(
            "ResNet-50/ImageNet official-baseline runs cannot load --resume; "
            "start from torchvision IMAGENET1K_V1 and use only the DSF artifact."
        )
    if not args.resume:
        return model, best_prec1

    if not os.path.isfile(args.resume):
        print(f"=> no checkpoint found at '{args.resume}'")
        return model, best_prec1

    print(f"=> loading checkpoint '{args.resume}'")
    checkpoint = torch.load(args.resume, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    filtered_state_dict, skipped = _filter_incompatible_state_dict(model, state_dict)
    has_shape_mismatch = any(new_shape is not None for _, _, new_shape in skipped)
    if skipped:
        print("=> skipped incompatible checkpoint tensors:")
        for key, old_shape, new_shape in skipped[:20]:
            print(f"   {key}: checkpoint={old_shape}, model={new_shape}")
        if len(skipped) > 20:
            print(f"   ... {len(skipped) - 20} more")
    load_state_dict_into_model(model, filtered_state_dict, strict=False)
    best_prec1 = checkpoint.get('best_prec1', checkpoint.get('prec1', 0.0))
    if has_shape_mismatch:
        print("=> checkpoint architecture differs from current model; reset best_prec1 to 0.0")
        best_prec1 = 0.0
    print(f"=> loaded checkpoint '{args.resume}'")
    return model, best_prec1


def save_baseline_checkpoint(model, args, prec1, prec5, epoch=-1, optimizer=None):
    if not is_main_process_local(args):
        return None

    baseline_ckpt_dir = os.path.join(args.save_dir, "baseline_ckpts")
    os.makedirs(baseline_ckpt_dir, exist_ok=True)
    base_model = unwrap_model(model)

    checkpoint = {
        'epoch': epoch,
        'best_prec1': float(prec1),
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'prec1': float(prec1),
        'prec5': float(prec5),
        'model': args.model,
        'dataset': args.dataset,
        'pretrained': bool(getattr(args, 'pretrained', False)),
        'num_classes': int(getattr(base_model.fc, 'out_features', 0)) if hasattr(base_model, 'fc') else None,
    }
    if is_resnet50_imagenet(args) and args.pretrained:
        checkpoint.update({
            'official_weights': OFFICIAL_RESNET50_V1_WEIGHTS,
            'official_baseline_sha256': OFFICIAL_RESNET50_V1_SHA256,
            'kd': False,
            'ema': False,
        })

    best_baseline_path = os.path.join(
        baseline_ckpt_dir,
        f"{args.model}_{args.dataset}_best_baseline.pth"
    )
    last_baseline_path = os.path.join(
        baseline_ckpt_dir,
        f"{args.model}_{args.dataset}_last_baseline.pth"
    )
    torch.save(checkpoint, best_baseline_path)
    torch.save(checkpoint, last_baseline_path)
    print(f'===> Saved baseline checkpoint to: {best_baseline_path}')
    return best_baseline_path


def set_train_stage(
    args,
    base_lr,
    total_epochs,
    lr_type=None,
    disable_lr_schedule=False,
    warm_epochs=None,
    min_lr=0.0,
):
    args.current_lr = float(base_lr)
    args.current_total_epochs = int(total_epochs)
    args.current_lr_type = lr_type if lr_type is not None else args.lr_type
    args.current_warm_epochs = 5 if warm_epochs is None else int(warm_epochs)
    args.current_min_lr = float(min_lr)
    args.disable_lr_schedule = bool(disable_lr_schedule)


def main():
    args = set_args()
    init_distributed_mode(args)
    print(args)

    os.makedirs(args.save_dir, exist_ok=True)
    save_run_config(args)

    device, data_parallel_device_ids = resolve_device_and_parallel_ids(args)

    seed = args.seed
    print('seed : ', seed)
    fix(seed)

    data_path = Path(args.data_dir).expanduser()
    if args.dataset == 'cifar10':
        num_classes, train_dataset, test_dataset = get_cifar10(data_path)
    elif args.dataset == 'imagenet':
        num_classes, train_dataset, test_dataset = get_imagenet(data_path)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    train_sampler = None
    val_sampler = None
    val_batch_size = 128
    if getattr(args, 'distributed', False):
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            test_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            drop_last=False,
        )
        val_batch_size = max(1, 128 // args.world_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda')
    )

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda')
    )

    print('train size:', len(train_dataset))
    print('val size:', len(test_dataset))
    if hasattr(train_dataset, 'classes'):
        print('train classes:', len(train_dataset.classes))
        print('first few classes:', train_dataset.classes[:10])

    baseline_label_smoothing = float(getattr(args, 'baseline_label_smoothing', 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=baseline_label_smoothing).to(device)

    model = build_model(args, num_classes, device)
    model, best_prec1 = maybe_load_resume(model, args, device)
    model = maybe_wrap_resnet50_imagenet_parallel(model, args, data_parallel_device_ids)
    print(model)

    print('===> baseline validate BEFORE compression/pruning')
    prec1, prec5 = validate(val_loader, model, criterion, args, device)
    initial_baseline_prec1 = float(prec1)
    if is_resnet50_imagenet(args) and args.pretrained and initial_baseline_prec1 < 75.9:
        raise RuntimeError(
            "Official ResNet-50 V1 baseline validation failed: "
            f"Top-1={initial_baseline_prec1:.3f}% < 75.900%"
        )
    initial_baseline_state = {
        key: value.detach().cpu().clone()
        for key, value in checkpoint_state_dict(model).items()
    }
    if args.save_eval_baseline:
        save_baseline_checkpoint(model, args, prec1, prec5, epoch=-1)
    if args.evaluate:
        print('===> evaluate only, skip training/compression/pruning.')
        cleanup_distributed(args)
        return

    # =========================
    # Base training
    # =========================
    # Configure the baseline learning-rate schedule.
    set_train_stage(
        args,
        base_lr=args.lr,
        total_epochs=args.epochs,
        lr_type=args.lr_type,
        disable_lr_schedule=False,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.current_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Keep the evaluated starting point as a valid best checkpoint so a
    # baseline finetune can never replace it with a worse epoch.
    best_prec1 = initial_baseline_prec1
    best_state = initial_baseline_state

    baseline_ckpt_dir = os.path.join(args.save_dir, "baseline_ckpts")
    os.makedirs(baseline_ckpt_dir, exist_ok=True)

    best_baseline_path = os.path.join(
        baseline_ckpt_dir,
        f"{args.model}_{args.dataset}_best_baseline.pth"
    )
    last_baseline_path = os.path.join(
        baseline_ckpt_dir,
        f"{args.model}_{args.dataset}_last_baseline.pth"
    )

    print(f'===> Train {args.model} on {args.dataset} for {args.epochs} epochs')
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args, device)
        prec1, prec5 = validate(val_loader, model, criterion, args, device)

        # 每轮都存 last，防止中断丢失
        if is_main_process_local(args):
            torch.save(
                {
                    'epoch': epoch,
                    'best_prec1': best_prec1,
                    'model_state_dict': checkpoint_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'prec1': prec1,
                    'prec5': prec5,
                    'model': args.model,
                    'dataset': args.dataset,
                },
                last_baseline_path,
            )

        # 保存 best baseline
        if float(prec1) > best_prec1:
            best_prec1 = float(prec1)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in checkpoint_state_dict(model).items()
            }

            save_baseline_checkpoint(
                model,
                args,
                prec1,
                prec5,
                epoch=epoch,
                optimizer=optimizer,
            )

            print(f'===> Saved BEST baseline to: {best_baseline_path}')

        print('best : ', best_prec1)

    if best_state is not None:
        load_state_dict_into_model(model, best_state, strict=True)
        print(f'===> Load best baseline checkpoint before DSF: {best_prec1:.4f}')

    dsf_channel_scores = None
    weff_mask_dict = None

    if args.load_dsf_artifacts:
        artifact_path = os.path.expanduser(args.load_dsf_artifacts)
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(f"--load-dsf-artifacts not found: {artifact_path}")
        artifact = torch.load(artifact_path, map_location='cpu')
        score_mode = str(getattr(args, 'a_row_norm_score', 'robust')).lower()
        score_variants = artifact.get('channel_score_variants', None)
        if isinstance(score_variants, dict) and score_mode in score_variants:
            dsf_channel_scores = score_variants[score_mode]
            print(f"===> Select DSF channel_score_variants['{score_mode}']")
        else:
            dsf_channel_scores = artifact.get('channel_scores', None)
            if isinstance(score_variants, dict):
                print(
                    f"[WARN] DSF artifact has score variants {sorted(score_variants.keys())}, "
                    f"but not '{score_mode}'; fallback to channel_scores."
                )
        mask_variants = artifact.get('weff_mask_variants', None)
        if isinstance(mask_variants, dict) and score_mode in mask_variants:
            weff_mask_dict = mask_variants[score_mode]
            print(f"===> Select DSF weff_mask_variants['{score_mode}']")
        else:
            weff_mask_dict = artifact.get('weff_mask_dict', None)
        if bool(getattr(args, 'load_dsf_model_state', False)):
            dsf_state = artifact.get('model_state_dict', None)
            if dsf_state is None:
                raise RuntimeError(f"--load-dsf-model-state requested, but no model_state_dict in: {artifact_path}")
            load_state_dict_into_model(model, dsf_state, strict=True)
            print(f"===> Loaded DSF model_state_dict from {artifact_path}")
        if dsf_channel_scores is None:
            raise RuntimeError(f"No channel_scores found in DSF artifact: {artifact_path}")
        print(
            f"===> Loaded DSF artifacts from {artifact_path}: "
            f"channel_scores={len(dsf_channel_scores)} "
            f"weff_masks={len(weff_mask_dict) if weff_mask_dict is not None else 0}"
        )

    if args.dsf:
        print('===> Start DSF (factorization + Lp/Lq regularization) ...')
        dsf_momentum = args.momentum if args.dsf_momentum is None else args.dsf_momentum
        dsf_weight_decay = args.weight_decay if args.dsf_weight_decay is None else args.dsf_weight_decay
        print(f'===> DSF recovery optimizer: lr={args.dsf_lr}, momentum={dsf_momentum}, weight_decay={dsf_weight_decay}')
        dsf_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.dsf_lr,
            momentum=dsf_momentum,
            weight_decay=dsf_weight_decay,
        )
        model, dsf_channel_scores, weff_mask_dict = compress_model(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=dsf_optimizer,
            epochs=args.compressed_epoch,
            device=device,
            args=args,
            val_loader=val_loader,
            sparsity=args.dsf_sparsity,
        )
        print('===> Finish DSF compression.')

    if args.sp:
        shape = (1, 3, 32, 32) if args.dataset in ('cifar10', 'cifar100') else (1, 3, 224, 224)
        sparse_model = get_channel_mag(
            model, shape, args, device,
            channel_score_dict=dsf_channel_scores,
            channel_mask_dict=weff_mask_dict,
            num_classes=num_classes,
        )
        if getattr(args, 'distributed', False):
            dist.barrier()
        sparse_model = sparse_model.to(device)
        sparse_model = maybe_wrap_resnet50_imagenet_parallel(
            sparse_model,
            args,
            data_parallel_device_ids,
        )
        _bn_calibrate(
            sparse_model,
            train_loader,
            device,
            args,
            num_batches=int(getattr(args, 'bn_calib_batches', 200)),
        )
        # Validate the actual compact model before any optimizer step.  This
        # catches channel/BN transfer errors separately from fine-tuning
        # instability and is enabled as a guard for ResNet-50/ImageNet runs.
        if is_resnet50_imagenet(args):
            sparse_init_prec1, sparse_init_prec5 = validate(
                val_loader, sparse_model, criterion, args, device
            )
            print(
                f'===> Structured-pruned init after BN calibration: '
                f'Prec@1={float(sparse_init_prec1):.4f}, '
                f'Prec@5={float(sparse_init_prec5):.4f}'
            )
            min_sparse_init = getattr(args, 'resnet50_min_sparse_init_prec1', None)
            if min_sparse_init is not None and float(sparse_init_prec1) < float(min_sparse_init):
                raise RuntimeError(
                    f'ResNet50/ImageNet structured-pruned init Prec@1='
                    f'{float(sparse_init_prec1):.3f} is below '
                    f'--resnet50-min-sparse-init-prec1={float(min_sparse_init):.3f}; '
                    'abort before long fine-tuning.'
                )
            if is_main_process_local(args):
                calibrated_init_path = os.path.join(
                    args.save_dir, 'finetune_model',
                    f'{args.model}_structured_pruned_init_bncalibrated.pth'
                )
                torch.save(
                    {
                        'model_state_dict': checkpoint_state_dict(sparse_model),
                        'prec1': float(sparse_init_prec1),
                        'prec5': float(sparse_init_prec5),
                        'model': args.model,
                        'dataset': args.dataset,
                    },
                    calibrated_init_path,
                )
                print(f'===> Saved calibrated structured-pruned init: {calibrated_init_path}')
        if (
            hasattr(torch, 'compile')
            and not isinstance(sparse_model, (nn.DataParallel, DDP))
            and not getattr(args, 'distributed', False)
        ):
            try:
                sparse_model = torch.compile(sparse_model)
            except Exception as exc:
                print(f'[WARN] torch.compile failed, continue without compile: {exc}')
        elif isinstance(sparse_model, (nn.DataParallel, DDP)) or getattr(args, 'distributed', False):
            print('[WARN] skip torch.compile for multi-GPU sparse model')

        # Use re_lr for the post-pruning learning-rate schedule.
        set_train_stage(
            args,
            base_lr=args.re_lr,
            total_epochs=args.fine_epochs,
            lr_type=args.re_lr_type,
            disable_lr_schedule=False,
            warm_epochs=args.sp_warm_epochs,
            min_lr=args.sp_min_lr,
        )

        sp_momentum = args.momentum if args.sp_momentum is None else args.sp_momentum
        sp_weight_decay = args.weight_decay if args.sp_weight_decay is None else args.sp_weight_decay
        optimizer = torch.optim.SGD(
            sparse_model.parameters(),
            lr=args.current_lr,
            momentum=sp_momentum,
            weight_decay=sp_weight_decay,
        )

        finetune_dir = os.path.join(args.save_dir, 'finetune_model')
        if is_main_process_local(args):
            os.makedirs(finetune_dir, exist_ok=True)
        best_sparse_prec1 = 0.0
        best_sparse_state = None
        torch.set_float32_matmul_precision('high')

        sp_label_smoothing = float(getattr(args, 'sp_label_smoothing', 0.1))
        print(
            f'===> Structured-pruning finetune: lr={args.re_lr}, lr_type={args.re_lr_type}, '
            f'epochs={args.fine_epochs}, label_smoothing={sp_label_smoothing}, '
            f'warm_epochs={getattr(args, "current_warm_epochs", 5)}, '
            f'min_lr={getattr(args, "current_min_lr", 0.0)}, '
            f'momentum={sp_momentum}, weight_decay={sp_weight_decay}')
        for epoch in range(args.fine_epochs):
            finetune_criterion = nn.CrossEntropyLoss(label_smoothing=sp_label_smoothing).to(device)
            train(train_loader, sparse_model, finetune_criterion, optimizer, epoch, args, device)
            prec1, prec5 = validate(val_loader, sparse_model, finetune_criterion, args, device)

            is_best_sparse = float(prec1) > best_sparse_prec1
            if is_best_sparse:
                best_sparse_prec1 = float(prec1)
                best_sparse_state = {
                    k: v.detach().cpu().clone()
                    for k, v in checkpoint_state_dict(sparse_model).items()
                }

            print('best : ', best_sparse_prec1)

            if is_main_process_local(args):
                payload = {
                    'epoch': epoch,
                    'model_state_dict': checkpoint_state_dict(sparse_model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'prec1': prec1,
                    'prec5': prec5,
                    'best_prec1': best_sparse_prec1,
                }
                torch.save(payload, os.path.join(finetune_dir, f'{args.model}_latest.pth'))
                if is_best_sparse:
                    torch.save(payload, os.path.join(finetune_dir, f'{args.model}_best.pth'))
                if should_save_epoch_checkpoint(epoch, args.fine_epochs, args.save_every):
                    torch.save(
                        payload,
                        os.path.join(finetune_dir, f'{args.model}_{epoch}.pth'),
                    )

        if best_sparse_state is not None:
            load_state_dict_into_model(sparse_model, best_sparse_state, strict=True)
            print(f'===> Load best sparse checkpoint: {best_sparse_prec1:.4f}')

        print('finish!')

    cleanup_distributed(args)


def train(train_loader, model, criterion, optimizer, epoch, args, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if hasattr(getattr(train_loader, 'sampler', None), 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)

    model.train()

    end = time.time()
    num_iter = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if not getattr(args, 'disable_lr_schedule', False):
            adjust_learning_rate(
                optimizer,
                epoch,
                i,
                num_iter,
                args,
                base_lr=getattr(args, 'current_lr', args.lr),
                total_epochs=getattr(args, 'current_total_epochs', args.epochs),
                lr_type=getattr(args, 'current_lr_type', args.lr_type),
            )

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        if args.half:
            input_var = input_var.half()

        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'LR {lr:.8f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), lr=current_lr,
                    batch_time=batch_time, data_time=data_time, loss=losses, top1=top1
                )
            )


if __name__ == '__main__':
    main()
