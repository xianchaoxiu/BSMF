import os
import json
import numpy as np
import torch
import torch.nn as nn
from select_channel import *
from compute_flops import *
from models.resnet import *
from models.hand_resnet50 import *
from models.built_model import VGG16CIFAR, VGG16_CIFAR_LAYOUT, vgg16_bn
import re
from thop import profile


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _is_main_process(args):
    return int(getattr(args, "rank", 0)) == 0


def _is_distributed(args):
    return bool(getattr(args, "distributed", False))


def _save_pruned_cfg(args, cfg, cfg_mask, num_classes):
    if not _is_main_process(args):
        return

    config_dir = os.path.join(getattr(args, "save_dir", "save_temp"), "config")
    os.makedirs(config_dir, exist_ok=True)

    cfg_list = [int(x) if isinstance(x, (np.integer, int)) else x for x in cfg]
    masks = []
    for mask in cfg_mask:
        bits = [int(v) for v in mask.detach().cpu().view(-1).tolist()]
        masks.append(
            {
                "bits": bits,
                "original_channels": len(bits),
                "retained_channels": int(sum(bits)),
                "retained_indices": [i for i, v in enumerate(bits) if v == 1],
            }
        )

    payload = {
        "model": getattr(args, "model", None),
        "dataset": getattr(args, "dataset", None),
        "num_classes": int(num_classes),
        "ratio": float(getattr(args, "ratio", 0.0)),
        "seed": int(getattr(args, "seed", 0)),
        "cfg": cfg_list,
        "cfg_length": len(cfg_list),
        "cfg_mask": masks,
    }

    json_path = os.path.join(config_dir, "pruned_cfg.json")
    txt_path = os.path.join(config_dir, "pruned_cfg.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("cfg = ")
        f.write(repr(cfg_list))
        f.write("\n")
        f.write("index,value\n")
        for i, value in enumerate(cfg_list):
            f.write(f"{i},{value}\n")

    print(f"===> Saved pruned cfg to: {json_path}")


cifar_cfg = {
    20: [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64],
    56: [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32,
         32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    44: [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
         32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    110: [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
          32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64,
          64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
          64]
}  # cfg lists output channel counts for the compact model.


def _parse_cifar_basicblock_name(layer_name):
    # 解析 CIFAR ResNet 的 BasicBlock 卷积层名字。
    # Layer-name examples:
        # layer1.block_0.conv_bn1.conv
        # layer1.block_0.conv_bn2.conv
    # 返回：
        # (stage_id, block_id, conv_id)
    # 例如：
        # layer3.block_17.conv_bn2.conv -> (3, 17, 2)
    m = re.search(r"layer(\d+)\.block_(\d+)\.conv_bn([12])\.conv", layer_name)
    if m is None:
        return None

    stage_id = int(m.group(1))
    block_id = int(m.group(2))
    conv_id = int(m.group(3))

    return stage_id, block_id, conv_id

def _is_imagenet_resnet50(args):
    return (
        str(getattr(args, "model", "")).lower() == "resnet50"
        and str(getattr(args, "dataset", "")).lower() == "imagenet"
    )


def _parse_resnet50_bottleneck_conv(layer_name):
    m = re.search(r"layer(\d+)\.(\d+)\.conv([123])$", layer_name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _is_resnet50_imagenet_sensitive(layer_name, args):
    if not _is_imagenet_resnet50(args):
        return False
    parsed = _parse_resnet50_bottleneck_conv(layer_name)
    if parsed is None:
        return False
    stage_id, block_id, conv_id = parsed
    if conv_id not in (1, 2):
        return False
    return (
        block_id == 0
        or (stage_id == 4 and block_id == 1)
    )


def _use_resnet50_layerwise_prune(args):
    return (
        _is_imagenet_resnet50(args)
        and bool(getattr(args, "resnet50_layerwise_prune", False))
    )


def _resnet50_layerwise_keep_ratio(layer_name, args):
    parsed = _parse_resnet50_bottleneck_conv(layer_name)
    if parsed is None:
        return None
    stage_id, block_id, conv_id = parsed
    if conv_id not in (1, 2):
        return None

    if block_id == 0:
        keep = float(getattr(args, "resnet50_keep_transition", 0.70))
    elif stage_id == 1:
        keep = float(getattr(args, "resnet50_keep_stage1", 0.75))
    elif stage_id == 2:
        keep = float(getattr(args, "resnet50_keep_stage2", 0.58))
    elif stage_id == 3:
        keep = float(getattr(args, "resnet50_keep_stage3", 0.42))
    elif stage_id == 4:
        keep = float(getattr(args, "resnet50_keep_stage4", 0.50))
    else:
        keep = 1.0

    if stage_id == 4 and block_id == 2:
        keep = max(keep, float(getattr(args, "resnet50_keep_last_block", 0.70)))
    if conv_id == 2:
        keep += float(getattr(args, "resnet50_conv2_keep_boost", 0.10))

    return min(1.0, max(0.01, keep))


def _resnet50_layerwise_keep_bounds(layer_name, args):
    keep = _resnet50_layerwise_keep_ratio(layer_name, args)
    if keep is None:
        return None
    margin = max(0.0, float(getattr(args, "resnet50_layerwise_bound_margin", 0.0)))
    min_ratio = max(0.01, keep - margin)
    max_ratio = min(1.0, keep + margin)
    return min_ratio, max(min_ratio, max_ratio)


def _is_resnet56_cifar(args):
    return (
        str(getattr(args, "model", "")).lower() == "resnet56"
        and str(getattr(args, "dataset", "")).lower() in ("cifar10", "cifar100")
    )


def _use_resnet56_layerwise_prune(args):
    return (
        _is_resnet56_cifar(args)
        and bool(getattr(args, "resnet56_layerwise_prune", False))
    )


def _use_resnet56_layerwise_bounds_only(args):
    return (
        _is_resnet56_cifar(args)
        and bool(getattr(args, "resnet56_layerwise_bounds_only", False))
    )


def _use_resnet56_adaptive_protect(args):
    return (
        _is_resnet56_cifar(args)
        and bool(getattr(args, "resnet56_adaptive_protect", False))
    )


def _resnet56_layerwise_keep_ratio(layer_name, args):
    parsed = _parse_cifar_basicblock_name(layer_name)
    if parsed is None:
        return None

    stage_id, block_id, conv_id = parsed
    if conv_id != 1:
        return None

    if stage_id == 1:
        keep = float(getattr(args, "resnet56_keep_stage1", 0.70))
    elif block_id == 0:
        keep = float(getattr(args, "resnet56_keep_transition", 0.65))
    elif stage_id == 2:
        keep = float(getattr(args, "resnet56_keep_stage2", 0.40))
    elif stage_id == 3:
        keep = float(getattr(args, "resnet56_keep_stage3", 0.30))
        tail_low_count = max(0, int(getattr(args, "resnet56_stage3_tail_low_count", 0)))
        tail_low_keep = getattr(args, "resnet56_stage3_tail_low_keep", None)
        if tail_low_count > 0 and tail_low_keep is not None:
            protected_count = max(0, int(getattr(args, "resnet56_last_block_count", 2)))
            first_tail_block = max(1, 9 - protected_count - tail_low_count)
            last_tail_block = max(0, 9 - protected_count - 1)
            if first_tail_block <= block_id <= last_tail_block:
                keep = float(tail_low_keep)
    else:
        keep = 1.0

    last_block_count = max(0, int(getattr(args, "resnet56_last_block_count", 2)))
    if stage_id == 3 and last_block_count > 0 and block_id >= 9 - last_block_count:
        keep = max(keep, float(getattr(args, "resnet56_keep_last_block", 0.48)))

    return min(1.0, max(0.01, keep))


def _resnet56_layerwise_keep_bounds(layer_name, args):
    keep = _resnet56_layerwise_keep_ratio(layer_name, args)
    if keep is None:
        return None

    parsed = _parse_cifar_basicblock_name(layer_name)
    margin = max(0.0, float(getattr(args, "resnet56_layerwise_bound_margin", 0.12)))

    min_ratio = max(0.01, keep - margin)
    max_ratio = min(1.0, keep + margin)

    if parsed is not None:
        stage_id, block_id, conv_id = parsed
        if conv_id == 1:
            if block_id == 0:
                min_ratio = max(min_ratio, float(getattr(args, "min_keep_sensitive", 0.50)))
                max_ratio = max(max_ratio, min(1.0, keep + margin * 0.75))
            elif block_id == 1:
                min_ratio = max(min_ratio, float(getattr(args, "min_keep_medium", 0.35)))
            if stage_id == 1:
                min_ratio = max(min_ratio, 0.50)
            last_block_count = max(0, int(getattr(args, "resnet56_last_block_count", 2)))
            if stage_id == 3 and last_block_count > 0 and block_id >= 9 - last_block_count:
                min_ratio = max(min_ratio, float(getattr(args, "resnet56_keep_last_block", 0.48)))

    min_ratio = min(1.0, max(0.01, min_ratio))
    max_ratio = min(1.0, max(min_ratio, max_ratio))
    return min_ratio, max_ratio


def _resnet56_base_floor_ratio(layer_name, args):
    parsed = _parse_cifar_basicblock_name(layer_name)
    if parsed is None:
        return None

    stage_id, block_id, conv_id = parsed
    if conv_id != 1:
        return None

    if block_id == 0:
        floor = float(getattr(args, "min_keep_sensitive", 0.50))
    elif block_id == 1:
        floor = float(getattr(args, "min_keep_medium", 0.35))
    elif stage_id == 1:
        floor = float(getattr(args, "min_keep_sensitive", 0.50))
    else:
        floor = float(getattr(args, "min_keep_normal", 0.15))

    last_block_count = max(0, int(getattr(args, "resnet56_last_block_count", 2)))
    if stage_id == 3 and last_block_count > 0 and block_id >= 9 - last_block_count:
        floor = max(floor, float(getattr(args, "resnet56_keep_last_block", 0.48)))

    return min(1.0, max(0.01, floor))


def _resnet56_adaptive_limits(layer_name, args):
    parsed = _parse_cifar_basicblock_name(layer_name)
    if parsed is None:
        return None

    stage_id, block_id, conv_id = parsed
    if conv_id != 1:
        return None

    min_override = None
    max_override = float(getattr(args, "resnet56_adaptive_max_keep", 1.0))

    if block_id not in (0, 1):
        if stage_id == 2:
            min_override = getattr(args, "resnet56_adaptive_min_stage2", None)
            stage_max = getattr(args, "resnet56_adaptive_max_stage2", None)
            if stage_max is not None:
                max_override = float(stage_max)
        elif stage_id == 3:
            min_override = getattr(args, "resnet56_adaptive_min_stage3", None)
            stage_max = getattr(args, "resnet56_adaptive_max_stage3", None)
            if stage_max is not None:
                max_override = float(stage_max)

    last_block_count = max(0, int(getattr(args, "resnet56_last_block_count", 2)))
    if stage_id == 3 and last_block_count > 0 and block_id >= 9 - last_block_count:
        last_max = getattr(args, "resnet56_adaptive_max_last_block", None)
        if last_max is not None:
            max_override = float(last_max)

    if min_override is not None:
        min_override = float(min_override)

    return min_override, min(1.0, max(0.01, float(max_override)))


def _resnet56_layerwise_sensitivity(layer_name, args):
    keep = _resnet56_layerwise_keep_ratio(layer_name, args)
    if keep is None:
        return None
    if keep >= 0.66:
        return "sensitive"
    if keep >= 0.50:
        return "medium"
    return "normal"


def _is_resnet110_cifar(args):
    return (
        str(getattr(args, "model", "")).lower() == "resnet110"
        and str(getattr(args, "dataset", "")).lower() in ("cifar10", "cifar100")
    )


def _use_resnet110_layerwise_prune(args):
    return (
        _is_resnet110_cifar(args)
        and bool(getattr(args, "resnet110_layerwise_prune", False))
    )


def _use_resnet110_layerwise_bounds_only(args):
    return (
        _is_resnet110_cifar(args)
        and bool(getattr(args, "resnet110_layerwise_bounds_only", False))
    )


def _resnet110_layerwise_keep_ratio(layer_name, args):
    parsed = _parse_cifar_basicblock_name(layer_name)
    if parsed is None:
        return None

    stage_id, block_id, conv_id = parsed
    if conv_id != 1:
        return None

    if block_id == 0:
        keep = float(getattr(args, "resnet110_keep_transition", 0.78))
    elif stage_id == 1:
        keep = float(getattr(args, "resnet110_keep_stage1", 0.52))
    elif stage_id == 2:
        keep = float(getattr(args, "resnet110_keep_stage2", 0.42))
    elif stage_id == 3:
        keep = float(getattr(args, "resnet110_keep_stage3", 0.40))
    else:
        keep = 1.0

    early_block_count = max(0, int(getattr(args, "resnet110_early_block_count", 2)))
    if early_block_count > 0 and block_id < early_block_count:
        keep = max(keep, float(getattr(args, "resnet110_keep_early_block", 0.62)))

    last_block_count = max(0, int(getattr(args, "resnet110_last_block_count", 2)))
    if stage_id == 3 and last_block_count > 0 and block_id >= 18 - last_block_count:
        keep = max(keep, float(getattr(args, "resnet110_keep_last_block", 0.52)))

    return min(1.0, max(0.01, keep))


def _resnet110_layerwise_keep_bounds(layer_name, args):
    keep = _resnet110_layerwise_keep_ratio(layer_name, args)
    if keep is None:
        return None

    margin = max(0.0, float(getattr(args, "resnet110_layerwise_bound_margin", 0.12)))
    min_ratio = max(0.01, keep - margin)
    max_ratio = min(1.0, keep + margin)
    return min_ratio, max_ratio


def _layerwise_sensitivity_from_keep(keep):
    if keep is None:
        return None
    if keep >= 0.66:
        return "sensitive"
    if keep >= 0.50:
        return "medium"
    return "normal"


def _resnet110_layerwise_sensitivity(layer_name, args):
    return _layerwise_sensitivity_from_keep(
        _resnet110_layerwise_keep_ratio(layer_name, args)
    )


# ----------------------------------------------------------------------
# VGG16 channel selection using A-row scores.
# ----------------------------------------------------------------------
_VGG16_BEST_WIDTHS = (48, 48, 97, 97, 196, 196, 196, 128, 128, 128, 128, 123, 512)
_VGG16_SPATIAL_SIZES = (32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2)


def _vgg16_macs(widths, num_classes):
    if len(widths) != len(_VGG16_SPATIAL_SIZES):
        raise ValueError(f"Expected 13 VGG16 widths, got {len(widths)}")

    total = 0.0
    in_channels = 3
    for out_channels, spatial in zip(widths, _VGG16_SPATIAL_SIZES):
        out_channels = int(out_channels)
        area = spatial * spatial
        total += area * out_channels * in_channels * 3 * 3
        total += 4 * area * out_channels
        in_channels = out_channels

    # HRank VGG uses AvgPool2d(2), then 512 -> 512 -> num_classes.
    total += int(widths[-1]) * 512
    total += 4 * 512
    total += 512 * int(num_classes)
    return float(total)


def _vgg16_params(widths, num_classes, classifier_hidden=512):
    """Count trainable parameters in the VGG16-CIFAR model."""
    if len(widths) != len(_VGG16_SPATIAL_SIZES):
        raise ValueError(f"Expected 13 VGG16 widths, got {len(widths)}")
    total = 0
    in_channels = 3
    for out_channels in widths:
        out_channels = int(out_channels)
        # Conv bias + BN affine parameters are included (1 + 2 per output).
        total += 9 * out_channels * in_channels + 3 * out_channels
        in_channels = out_channels
    hidden = int(classifier_hidden)
    # Linear1 bias + BN1d affine + Linear2 bias.
    total += int(widths[-1]) * hidden + 3 * hidden + hidden * int(num_classes) + int(num_classes)
    return int(total)


def _vgg16_copy_pruned_weights(source_model, sparse_model, masks):
    source_features = list(source_model.features)
    sparse_features = list(sparse_model.features)
    conv_positions = [
        index for index, module in enumerate(source_features) if isinstance(module, nn.Conv2d)
    ]

    input_indices = torch.arange(
        3, device=source_model.classifier.linear1.weight.device
    )
    for layer_index, feature_index in enumerate(conv_positions):
        source_conv = source_features[feature_index]
        sparse_conv = sparse_features[feature_index]
        output_indices = torch.nonzero(masks[layer_index] > 0.5, as_tuple=False).flatten()

        sparse_conv.weight.data.copy_(
            source_conv.weight.data[output_indices][:, input_indices].clone()
        )
        if source_conv.bias is not None:
            sparse_conv.bias.data.copy_(source_conv.bias.data[output_indices].clone())

        source_bn = source_features[feature_index + 1]
        sparse_bn = sparse_features[feature_index + 1]
        sparse_bn.weight.data.copy_(source_bn.weight.data[output_indices].clone())
        sparse_bn.bias.data.copy_(source_bn.bias.data[output_indices].clone())
        sparse_bn.running_mean.data.copy_(source_bn.running_mean.data[output_indices].clone())
        sparse_bn.running_var.data.copy_(source_bn.running_var.data[output_indices].clone())
        sparse_bn.num_batches_tracked.data.copy_(source_bn.num_batches_tracked.data)
        input_indices = output_indices

    hidden = sparse_model.classifier.linear1.out_features
    source_hidden = source_model.classifier.linear1.out_features
    source_input = source_model.classifier.linear1.weight.data[:, input_indices]
    if hidden == source_hidden:
        hidden_indices = torch.arange(source_hidden, device=source_input.device)
    else:
        hidden_score = torch.linalg.vector_norm(source_input, ord=2, dim=1)
        hidden_indices = torch.topk(hidden_score, k=hidden, largest=True).indices
    sparse_model.classifier.linear1.weight.data.copy_(source_input[hidden_indices].clone())
    sparse_model.classifier.linear1.bias.data.copy_(source_model.classifier.linear1.bias.data[hidden_indices].clone())
    sparse_model.classifier.norm1.weight.data.copy_(source_model.classifier.norm1.weight.data[hidden_indices].clone())
    sparse_model.classifier.norm1.bias.data.copy_(source_model.classifier.norm1.bias.data[hidden_indices].clone())
    sparse_model.classifier.norm1.running_mean.data.copy_(source_model.classifier.norm1.running_mean.data[hidden_indices].clone())
    sparse_model.classifier.norm1.running_var.data.copy_(source_model.classifier.norm1.running_var.data[hidden_indices].clone())
    sparse_model.classifier.norm1.num_batches_tracked.data.copy_(source_model.classifier.norm1.num_batches_tracked.data)
    sparse_model.classifier.linear2.weight.data.copy_(source_model.classifier.linear2.weight.data[:, hidden_indices].clone())
    sparse_model.classifier.linear2.bias.data.copy_(source_model.classifier.linear2.bias.data.clone())


def _vgg16_select_masks(model, channel_scores, widths, strict=True):
    convs = [(f'features.{name}', module)
             for name, module in model.features.named_children()
             if isinstance(module, nn.Conv2d)]
    if len(convs) != 13 or len(widths) != 13:
        raise ValueError('VGG16 requires exactly 13 convolution widths and scores.')
    masks = []
    for (name, conv), width in zip(convs, widths):
        if isinstance(width, bool) or int(width) != width or not 1 <= int(width) <= conv.out_channels:
            raise ValueError(f'{name}: invalid keep width {width} / {conv.out_channels}')
        if channel_scores is not None and name in channel_scores:
            score = torch.as_tensor(channel_scores[name]).detach().float().to(conv.weight.device)
        elif strict:
            raise RuntimeError(f'Missing VGG16 A-row scores for {name}; use --dsf to compute channel scores.')
        else:
            score = torch.linalg.vector_norm(conv.weight.detach().flatten(1), dim=1)
        if score.ndim != 1 or score.numel() != conv.out_channels or not torch.isfinite(score).all():
            raise ValueError(f'{name}: expected {conv.out_channels} finite channel scores.')
        # Apply sorted top-k to the normalized A-row scores.
        # Quantile clipping can create ties, so keep the scores and ranking unchanged.
        indices = torch.topk(score, k=int(width), largest=True).indices
        mask = torch.zeros_like(score)
        mask[indices] = 1.0
        masks.append(mask)
    return masks


def _get_channel_mag_vgg16(model, shape, args, device, channel_score_dict=None,
                         channel_mask_dict=None, num_classes=None):
    if (str(args.model).lower() != 'vgg16'
            or str(args.dataset).lower() not in ('cifar10', 'cifar100')):
        raise ValueError('VGG16 pruning is restricted to the VGG16/CIFAR branch.')
    if tuple(shape[-3:]) != (3, 32, 32) or not isinstance(model, VGG16CIFAR):
        raise ValueError('VGG16 pruning requires the 32x32 VGG16CIFAR architecture.')
    num_classes = int(num_classes or model.classifier.linear2.out_features)
    if model.classifier.linear1.out_features != 512:
        raise ValueError('VGG16 pruning requires a classifier hidden width of 512.')
    widths = list(getattr(args, 'vgg16_channel_widths', None) or _VGG16_BEST_WIDTHS)
    masks = _vgg16_select_masks(model, channel_score_dict, widths,
                               strict=bool(getattr(args, 'strict_a_row_norm', True)))
    base_widths = [int(v) for v in VGG16_CIFAR_LAYOUT if v != 'M']
    base_macs, pruned_macs = _vgg16_macs(base_widths, num_classes), _vgg16_macs(widths, num_classes)
    base_params, pruned_params = _vgg16_params(base_widths, num_classes), _vgg16_params(widths, num_classes)
    flops_drop = 100.0 * (1.0 - pruned_macs / base_macs)
    params_drop = 100.0 * (1.0 - pruned_params / base_params)
    for flag, target, actual in (
            ('--target-flops-drop', getattr(args, 'target_flops_drop', None), flops_drop),
            ('--vgg16-target-params-drop', getattr(args, 'vgg16_target_params_drop', None), params_drop)):
        if target is not None:
            target = float(target) * (100.0 if float(target) <= 1.0 else 1.0)
            if not 0.0 <= target < 100.0:
                raise ValueError(f'{flag} must be in [0, 100).')
            # Resource budgets specify minimum reductions.
            if actual + 1e-8 < target:
                raise ValueError(f'VGG16 widths give {actual:.6f}% reduction, below {flag}={target:.6f}%.')
    sparse_model = vgg16_bn(cfg=widths, num_classes=num_classes).to(device)
    _vgg16_copy_pruned_weights(model, sparse_model, masks)
    if sum(p.numel() for p in sparse_model.parameters()) != pruned_params:
        raise RuntimeError('VGG16 parameter count disagrees with the pruning configuration.')
    args.actual_flops_drop, args.actual_params_drop = flops_drop, params_drop
    args.pruned_macs, args.base_macs = pruned_macs, base_macs
    args.pruned_params, args.base_params = pruned_params, base_params
    if _is_main_process(args):
        _save_pruned_cfg(args, widths, masks, num_classes)
        config_dir = os.path.join(args.save_dir, 'config')
        payload = {
            'model': 'vgg16', 'dataset': args.dataset, 'num_classes': num_classes,
            'channel_widths': widths, 'classifier_hidden': 512,
            'a_row_norm_score': getattr(args, 'a_row_norm_score', 'robust'),
            'correlation_penalty': 0.0, 'actual_flops_drop': flops_drop,
            'actual_params_drop': params_drop, 'base_macs': base_macs,
            'pruned_macs': pruned_macs, 'base_params': base_params,
            'pruned_params': pruned_params,
            'masks': [mask.detach().cpu().int().tolist() for mask in masks],
        }
        with open(os.path.join(config_dir, 'vgg16_pruned_cfg.json'), 'w') as handle:
            json.dump(payload, handle, indent=2)
        init_dir = os.path.join(args.save_dir, 'finetune_model')
        os.makedirs(init_dir, exist_ok=True)
        torch.save({
            'model_state_dict': sparse_model.state_dict(), 'cfg': widths,
            'cfg_mask': [mask.detach().cpu() for mask in masks],
            'model': 'vgg16', 'dataset': args.dataset, 'num_classes': num_classes,
            'actual_flops_drop': flops_drop, 'post_pruning_epochs_inherited_by_weights': 0,
        }, os.path.join(init_dir, 'vgg16_structured_pruned_init.pth'))
    print(f'[VGG16-Pruning] widths={widths}, MACs={pruned_macs:.0f}, params={pruned_params}, '
          f'FLOPs drop={flops_drop:.6f}%, params drop={params_drop:.6f}%')
    return sparse_model


def get_channel_mag(model, shape, args, device, channel_score_dict=None, channel_mask_dict=None, num_classes=None):  ##按通道重要性（magnitude）剪枝的主函数
    model = _unwrap_model(model)
    if str(args.model).lower() == 'vgg16':
        return _get_channel_mag_vgg16(model, shape, args, device, channel_score_dict,
                                     channel_mask_dict, num_classes)

    if num_classes is None:
        if args.dataset == "cifar10":
            num_classes = 10
        elif args.dataset == "cifar100":
            num_classes = 100
        elif args.dataset == "tiny_imagenet":
            num_classes = 200
        elif args.dataset == "imagenet":
            # ImageNet experiments can be subsets or the full 1K dataset.
            # Infer the active head size from the loaded model whenever possible.
            fc = getattr(model, "fc", None)
            num_classes = int(getattr(fc, "out_features", 1000))
        else:
            raise ValueError

    def _get_layer_channel_importance(module, layer_name, require_a_row_norm=False):
        fallback = get_channel_L2(module).abs()
        strict_a_row_norm = bool(getattr(args, "strict_a_row_norm", True))

        if channel_score_dict is None:
            if strict_a_row_norm and require_a_row_norm:
                raise RuntimeError(
                    f"Missing DSF A-row-norm scores for prunable layer {layer_name}. "
                    "Run with --dsf so compress_model can export ||A_i||_2 scores, "
                    "or pass --no-strict-a-row-norm to allow fallback scores."
                )
            return fallback

        if layer_name not in channel_score_dict:
            if strict_a_row_norm and require_a_row_norm:
                raise RuntimeError(
                    f"Layer {layer_name} has no DSF A-row-norm score. "
                    "Structured pruning is configured to use A-row norms only."
                )
            return fallback

        score = channel_score_dict[layer_name].to(device).float()

        # 长度保护：score 必须等于当前卷积层 out_channels
        if score.numel() != module.weight.data.shape[0]:
            msg = (
                f"score size mismatch for {layer_name}: "
                f"score={score.numel()} vs out_channels={module.weight.data.shape[0]}"
            )
            if strict_a_row_norm and require_a_row_norm:
                raise RuntimeError(f"{msg}. Cannot prune without valid A-row-norm scores.")
            print(f"[WARN] {msg}, fallback to magnitude score.")
            return fallback

        return score

    def _repair_mask_with_min_keep(weight_copy, mask, min_keep):
        remain = int(torch.sum(mask).item())
        if remain >= min_keep:
            return mask, remain

        # 按分数强制保留 top-k
        top_values, top_indices = torch.topk(weight_copy, k=min_keep, largest=True)
        new_mask = torch.zeros_like(weight_copy)
        new_mask[top_indices] = 1.0
        return new_mask, min_keep

    def _repair_mask_with_keep_bounds(weight_copy, mask, min_keep, max_keep):
        mask, remain = _repair_mask_with_min_keep(weight_copy, mask, min_keep)
        if remain <= max_keep:
            return mask, remain

        _, top_indices = torch.topk(weight_copy, k=max_keep, largest=True)
        new_mask = torch.zeros_like(weight_copy)
        new_mask[top_indices] = 1.0
        return new_mask, max_keep

    def _estimate_spatial_area(layer_name):
        # 近似估计每个 stage 的空间尺寸，用于 FLOPs-aware 排序
        if args.dataset in ['cifar10', 'cifar100']:
            if 'layer1' in layer_name:
                return 32 * 32
            elif 'layer2' in layer_name:
                return 16 * 16
            elif 'layer3' in layer_name:
                return 8 * 8
            else:
                return 32 * 32

        # ImageNet / Tiny-ImageNet 风格的大图输入
        if 'layer1' in layer_name:
            return 56 * 56
        elif 'layer2' in layer_name:
            return 28 * 28
        elif 'layer3' in layer_name:
            return 14 * 14
        elif 'layer4' in layer_name:
            return 7 * 7
        else:
            return 56 * 56


    def _make_prune_score(score, module, layer_name):
        # score: A 行范数分数 / channel_score_dict 分数
        # 返回用于全局阈值排序的分数。
        # 分数越小，越容易被剪。
        score = score.detach().float().to(device)

        if not getattr(args, "flops_aware", False):
            return score

        if bool(getattr(args, "strict_a_row_norm", True)):
            # FLOPs 目标只作为预算约束；通道评价仍严格由 A 的行范数决定。
            return score

        k_ops = module.kernel_size[0] * module.kernel_size[1]
        cin = module.in_channels / module.groups
        spatial = _estimate_spatial_area(layer_name)

        # 每个输出通道的大致 FLOPs 代价
        cost = float(k_ops * cin * spatial)

        # 防止 cost 过大导致分数尺度崩坏
        cost_norm = max(cost / 10000.0, 1.0)
        alpha = float(getattr(args, "flops_alpha", 0.25))

        return score / (cost_norm ** alpha)

    def _use_fixed_dsf_mask(layer_name):
        return (
            bool(getattr(args, "use_dsf_hard_mask", True))
            and channel_mask_dict is not None
            and layer_name in channel_mask_dict
        )

    def _append_cfg_entry(value, mask, layer_name, is_main_cfg=True):
        cfg_mask.append(mask)
        if args.model == 'resnet50' and not is_main_cfg:
            return
        cfg.append(value)

    def _is_prunable_conv_index(conv_index):
        if args.model == 'vgg16':
            return True
        return conv_index not in conv_locate and conv_index not in down_locate

    def _normalized_target_flops_drop():
        target = getattr(args, "target_flops_drop", None)
        if target is None:
            return None
        target = float(target)
        if target <= 1.0:
            target *= 100.0
        if target < 0.0 or target >= 100.0:
            raise ValueError(f"--target-flops-drop must be in [0, 100), got {target}")
        return target

    _resnet50_imagenet_profile_cache = {}

    def _resnet50_imagenet_profile_from_cfg(cfg_values):
        if args.model != 'resnet50' or args.dataset != 'imagenet':
            return None

        cache_key = None if cfg_values is None else tuple(int(v) for v in cfg_values)
        if cache_key in _resnet50_imagenet_profile_cache:
            return _resnet50_imagenet_profile_cache[cache_key]

        try:
            if cfg_values is None:
                flops_model = pruned_resnet50(dataset=args.dataset, num_classes=num_classes)
            else:
                cfg_list = [int(v) for v in cfg_values]
                if len(cfg_list) not in (49, 50):
                    print(
                        f"[Target-FLOPs] invalid ResNet-50 cfg length: {len(cfg_list)} "
                        f"(expected 49 or generated 50)"
                    )
                    return None
                flops_model = pruned_resnet50(cfg=cfg_list, dataset=args.dataset, num_classes=num_classes)

            flops_model = flops_model.cpu()
            flops_model.eval()
            input_data = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                flops, params = profile(flops_model, (input_data,), verbose=False)
            result = (float(flops), float(params))
            _resnet50_imagenet_profile_cache[cache_key] = result
            return result
        except Exception as exc:
            print(f"[Target-FLOPs] ResNet-50/ImageNet FLOPs profiling failed: {exc}")
            return None

    def _resnet50_imagenet_flops_from_cfg(cfg_values):
        result = _resnet50_imagenet_profile_from_cfg(cfg_values)
        if result is None:
            return None
        return result[0]

    def _resnet50_imagenet_cfg_from_records(layer_records):
        remain_by_name = {
            record["name"]: int(record["mask"].sum().item())
            for record in layer_records
        }
        stem = remain_by_name.get("conv1", 64)
        stage_blocks = {1: 3, 2: 4, 3: 6, 4: 3}
        stage_out = {1: 256, 2: 512, 3: 1024, 4: 2048}
        default_mid = {1: 64, 2: 128, 3: 256, 4: 512}

        cfg_values = [int(stem)]
        input_width = int(stem)
        for stage_id in (1, 2, 3, 4):
            for block_id in range(stage_blocks[stage_id]):
                conv1_name = f"layer{stage_id}.{block_id}.conv1"
                conv2_name = f"layer{stage_id}.{block_id}.conv2"
                cfg_values.extend(
                    [
                        int(input_width),
                        int(remain_by_name.get(conv1_name, default_mid[stage_id])),
                        int(remain_by_name.get(conv2_name, default_mid[stage_id])),
                    ]
                )
                input_width = int(stage_out[stage_id])
        return cfg_values

    def _cifar_resnet_flops_from_cfg(cfg_values):
        if args.model == 'resnet56':
            depth = 56
        elif args.model == 'resnet110':
            depth = 110
        else:
            return None
        if args.dataset not in ['cifar10', 'cifar100']:
            return None

        num_blocks = (depth - 2) // 6
        convs_per_stage = num_blocks * 2
        spatial_sizes = [32, 16, 8]
        total_flops = 0.0

        total_flops += 3 * int(cfg_values[0]) * 3 * 3 * 32 * 32
        total_flops += 2 * int(cfg_values[0]) * 32 * 32

        for stage_id, spatial in enumerate(spatial_sizes):
            stage_start = stage_id * convs_per_stage
            for offset in range(0, convs_per_stage, 2):
                in_channels = int(cfg_values[stage_start + offset])
                mid_channels = int(cfg_values[stage_start + offset + 1])
                out_channels = int(cfg_values[stage_start + offset + 2])

                total_flops += in_channels * mid_channels * 3 * 3 * spatial * spatial
                total_flops += 2 * mid_channels * spatial * spatial
                total_flops += mid_channels * out_channels * 3 * 3 * spatial * spatial
                total_flops += 2 * out_channels * spatial * spatial

        total_flops += int(cfg_values[-1]) * int(num_classes) + int(num_classes)
        return float(total_flops)

    def _cfg_from_records(layer_records):
        if args.model == 'resnet50' and args.dataset == 'imagenet':
            return _resnet50_imagenet_cfg_from_records(layer_records)
        cfg_values = []
        for record in layer_records:
            cfg_values.append(int(record["mask"].sum().item()))
        return cfg_values

    def _estimate_flops_drop_from_records(layer_records):
        current_cfg = _cfg_from_records(layer_records)
        base_cfg = None
        if args.model in ['resnet56', 'resnet110']:
            depth = int(str(args.model).replace('resnet', ''))
            base_cfg = cifar_cfg[depth]
        elif args.model == 'resnet50' and args.dataset == 'imagenet':
            base_flops = _resnet50_imagenet_flops_from_cfg(None)
            current_flops = _resnet50_imagenet_flops_from_cfg(current_cfg)
            if base_flops is None or current_flops is None:
                return None, None, None
            drop = 100.0 - current_flops / base_flops * 100.0
            return drop, current_flops, base_flops
        if base_cfg is None:
            return None, None, None

        base_flops = _cifar_resnet_flops_from_cfg(base_cfg)
        current_flops = _cifar_resnet_flops_from_cfg(current_cfg)
        if base_flops is None or current_flops is None:
            return None, None, None
        drop = 100.0 - current_flops / base_flops * 100.0
        return drop, current_flops, base_flops

    def _sync_records_to_outputs(layer_records):
        cfg[:] = []
        cfg_mask[:] = []
        pruned_count = 0

        for record in layer_records:
            mask = record["mask"].detach().float().to(device)
            record["mask"] = mask
            remain = int(mask.sum().item())
            record["remain"] = remain
            cfg_mask.append(mask.clone())
            if args.model != 'resnet50':
                cfg.append(remain)
            if record.get("is_prunable", False):
                pruned_count += int(mask.numel() - remain)

        if args.model == 'resnet50' and args.dataset == 'imagenet':
            cfg[:] = _resnet50_imagenet_cfg_from_records(layer_records)

        return pruned_count

    def _adjust_records_to_target_flops(layer_records):
        target_drop = _normalized_target_flops_drop()
        if target_drop is None:
            return None
        if (
            (args.model in ['resnet56', 'resnet110'] and args.dataset not in ['cifar10', 'cifar100'])
            or (args.model == 'resnet50' and args.dataset != 'imagenet')
            or args.model not in ['resnet56', 'resnet110', 'resnet50']
        ):
            print(f"[Target-FLOPs] skip exact target for {args.model}/{args.dataset}")
            return None

        tolerance = max(0.0, float(getattr(args, "target_flops_tolerance", 0.05)))
        best_records = None
        best_error = None
        max_iters = 10000
        if args.model == 'resnet50':
            max_iters = 2000

        for _ in range(max_iters):
            drop, current_flops, base_flops = _estimate_flops_drop_from_records(layer_records)
            if drop is None:
                return None
            error = drop - target_drop
            if best_error is None or abs(error) < abs(best_error):
                best_error = error
                best_records = [record["mask"].detach().clone() for record in layer_records]
            if abs(error) <= tolerance:
                print(
                    f"[Target-FLOPs] reached target={target_drop:.3f}% "
                    f"actual={drop:.3f}% flops={current_flops:.0f}/{base_flops:.0f}"
                )
                return drop

            candidates = []
            need_more_drop = error < 0.0
            for record_index, record in enumerate(layer_records):
                if not record.get("is_prunable", False):
                    continue
                score = record["score"].detach().float().to(device)
                mask = record["mask"].detach().float().to(device)
                remain = int(mask.sum().item())

                if need_more_drop:
                    if remain <= int(record["min_keep"]):
                        continue
                    available = torch.nonzero(mask > 0.5, as_tuple=False).view(-1)
                    if available.numel() == 0:
                        continue
                    local_scores = score[available]
                    pos = int(torch.argmin(local_scores).item())
                    channel_index = int(available[pos].item())
                    priority = float(local_scores[pos].item())
                    candidates.append((priority, record_index, channel_index))
                else:
                    if remain >= int(record["max_keep"]):
                        continue
                    available = torch.nonzero(mask <= 0.5, as_tuple=False).view(-1)
                    if available.numel() == 0:
                        continue
                    local_scores = score[available]
                    pos = int(torch.argmax(local_scores).item())
                    channel_index = int(available[pos].item())
                    priority = -float(local_scores[pos].item())
                    candidates.append((priority, record_index, channel_index))

            if not candidates:
                print(
                    f"[Target-FLOPs] no movable A-row-norm channels left; "
                    f"target={target_drop:.3f}% best_error={best_error:.3f}%"
                )
                break

            _priority, record_index, channel_index = sorted(candidates, key=lambda x: x[0])[0]
            layer_mask = layer_records[record_index]["mask"].detach().clone().to(device)
            layer_mask[channel_index] = 0.0 if need_more_drop else 1.0
            layer_records[record_index]["mask"] = layer_mask

        if best_records is not None:
            for record, best_mask in zip(layer_records, best_records):
                record["mask"] = best_mask.to(device)
            drop, current_flops, base_flops = _estimate_flops_drop_from_records(layer_records)
            print(
                f"[Target-FLOPs] use closest target={target_drop:.3f}% "
                f"actual={drop:.3f}% error={drop - target_drop:.3f}% "
                f"flops={current_flops:.0f}/{base_flops:.0f}"
            )
            return drop
        return None

    total = 0  # Total prunable output channels.
    local = 0
    conv_local = 1  # One-based convolution index.
    down_locate = []  # Indices of shortcut projection convolutions.
    if args.model != 'vgg16':
        conv_locate = []  # Protected convolution indices.
        conv_locate.append(1)  # Protect the stem convolution.
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):

                print('name : ', name)
                if args.model == 'resnet56' or args.model == 'resnet110':
                    if 'conv_bn2' in name:
                        conv_locate.append(conv_local)
                    conv_local += 1
                elif args.model == 'resnet50':
                    if 'conv3' in name:  ##不剪
                        conv_locate.append(conv_local)
                    if 'layer4.2' in name and not _use_resnet50_layerwise_prune(args):  ##不剪
                        conv_locate.append(conv_local)
                    if 'downsample' in name:  ##不剪
                        down_locate.append(conv_local)
                    conv_local += 1

    conv_local = 1
    print('conv locate : ', conv_locate)
    for m in model.modules():

        if isinstance(m, nn.Conv2d):
            if args.model != 'vgg16' and conv_local in conv_locate:
                conv_local += 1
                continue
            if args.model != 'vgg16' and conv_local in down_locate:
                conv_local += 1
                continue
            conv_local += 1
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)  # Concatenated channel scores.
    local = 0
    index = 0
    conv_local = 1
    sensitive_layer1 = []
    sensitive_layer = []
    medium_sensitive_layer = []

    if args.model in ['resnet56', 'resnet110']:
        conv_local = 1
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 只保护真正参与剪枝的 conv_bn1
                if 'conv_bn1' in name:
                    layerwise_sensitivity = None
                    if _use_resnet56_layerwise_bounds_only(args):
                        layerwise_sensitivity = _resnet56_layerwise_sensitivity(name, args)
                    elif _use_resnet110_layerwise_prune(args) or _use_resnet110_layerwise_bounds_only(args):
                        layerwise_sensitivity = _resnet110_layerwise_sensitivity(name, args)

                    if layerwise_sensitivity == "sensitive":
                        sensitive_layer.append(conv_local)
                    elif layerwise_sensitivity == "medium":
                        medium_sensitive_layer.append(conv_local)
                    elif (
                            'layer1.block_0' in name
                            or 'layer2.block_0' in name
                            or 'layer3.block_0' in name
                    ):
                        sensitive_layer.append(conv_local)
                    elif (
                            'layer1.block_1' in name
                            or 'layer2.block_1' in name
                            or 'layer3.block_1' in name
                    ):
                        medium_sensitive_layer.append(conv_local)
                conv_local += 1

    elif args.model == 'resnet50':
        conv_local = 1
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if _use_resnet50_layerwise_prune(args):
                    keep_ratio = _resnet50_layerwise_keep_ratio(name, args)
                    if keep_ratio is not None:
                        if keep_ratio >= 0.70:
                            sensitive_layer.append(conv_local)
                        elif keep_ratio >= 0.55:
                            medium_sensitive_layer.append(conv_local)
                else:
                    if 'layer2.0' in name or 'layer3.0' in name:
                        if 'conv1' in name or 'conv2' in name:
                            sensitive_layer.append(conv_local)
                    if 'layer4.1' in name or 'layer4.0' in name:
                        sensitive_layer.append(conv_local)
                    if 'layer1.0' in name:
                        sensitive_layer.append(conv_local)
                conv_local += 1

    print('sensitive layer : ', sensitive_layer)
    print('medium sensitive layer : ', medium_sensitive_layer)
    print('downsample : ', down_locate)

    resnet56_stage_score_mean = {}
    if _use_resnet56_adaptive_protect(args):
        score_buckets = {}
        for lname, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            parsed = _parse_cifar_basicblock_name(lname)
            if parsed is None:
                continue
            stage_id, _block_id, conv_id = parsed
            if conv_id != 1:
                continue
            score = _get_layer_channel_importance(
                module,
                lname,
                require_a_row_norm=True,
            ).detach().float()
            score_buckets.setdefault(stage_id, []).append(float(score.mean().item()))
        for stage_id, values in score_buckets.items():
            if values:
                resnet56_stage_score_mean[stage_id] = float(np.mean(values))

    conv_local = 1
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if args.model != 'vgg16' and conv_local in conv_locate:
                conv_local += 1
                continue
            if args.model != 'vgg16' and conv_local in down_locate:
                conv_local += 1
                continue

            size = m.weight.data.shape[0]
            channel_importance = _get_layer_channel_importance(
                m,
                name,
                require_a_row_norm=True,
            )
            prune_importance = _make_prune_score(channel_importance, m, name)

            conv_local += 1
            bn[index:(index + size)] = prune_importance.clone()
            index += size

    bn1 = bn
    y, i = torch.sort(bn)  ##对汇总后的分数做升序排序。
    thre_index = min(int(total * args.ratio), total - 1)
    # determine threshold
    thre = y[thre_index]
    print('thre : ', thre)
    pruned = 0  ##计数总共剪掉了多少个通道
    cfg = []  # Retained channel counts and masks for compact-model construction.
    cfg_mask = []

    layer_records = []

    local = 0  ##重置各种指针/层编号，方便下面再次遍历网络时对齐“第几层卷积”
    conv_local = 1

    for k, (name, m) in enumerate(model.named_modules()):

        if isinstance(m, nn.Conv2d):
            if args.model != 'vgg16' and conv_local in conv_locate:
                current_conv_name = name
                conv_local += 1
                if args.model == 'resnet56' or args.model == 'resnet110':
                    weight_copy = _get_layer_channel_importance(m, name)
                    mask = torch.ones_like(weight_copy).to(device)
                    layer_records.append(
                        {
                            "name": name,
                            "mask": mask.clone(),
                            "score": weight_copy.detach().float().to(device),
                            "weight": m.weight.data.detach().clone(),
                            "is_prunable": False,
                            "is_main_cfg": True,
                            "min_keep": int(mask.numel()),
                            "max_keep": int(mask.numel()),
                        }
                    )

                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                          format(k, weight_copy.numel(), weight_copy.numel()))
                    continue
                elif args.model == 'resnet50':
                    weight_copy = _get_layer_channel_importance(m, name)
                    mask = torch.ones_like(weight_copy).to(device)
                    layer_records.append(
                        {
                            "name": name,
                            "mask": mask.clone(),
                            "score": weight_copy.detach().float().to(device),
                            "weight": m.weight.data.detach().clone(),
                            "is_prunable": False,
                            "is_main_cfg": True,
                            "min_keep": int(mask.numel()),
                            "max_keep": int(mask.numel()),
                        }
                    )

                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                          format(k, weight_copy.numel(), weight_copy.numel()))
                    continue

            if args.model != 'vgg16' and conv_local in down_locate:
                current_conv_name = name
                conv_local += 1
                if args.model == 'resnet56' or args.model == 'resnet110':
                    weight_copy = _get_layer_channel_importance(m, name)
                    mask = torch.ones_like(weight_copy).to(device)
                    layer_records.append(
                        {
                            "name": name,
                            "mask": mask.clone(),
                            "score": weight_copy.detach().float().to(device),
                            "weight": m.weight.data.detach().clone(),
                            "is_prunable": False,
                            "is_main_cfg": True,
                            "min_keep": int(mask.numel()),
                            "max_keep": int(mask.numel()),
                        }
                    )

                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                          format(k, weight_copy.numel(), weight_copy.numel()))
                    continue
                elif args.model == 'resnet50':
                    weight_copy = _get_layer_channel_importance(m, name)
                    mask = torch.ones_like(weight_copy).to(device)
                    layer_records.append(
                        {
                            "name": name,
                            "mask": mask.clone(),
                            "score": weight_copy.detach().float().to(device),
                            "weight": m.weight.data.detach().clone(),
                            "is_prunable": False,
                            "is_main_cfg": False,
                            "min_keep": int(mask.numel()),
                            "max_keep": int(mask.numel()),
                        }
                    )

                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                          format(k, weight_copy.numel(), weight_copy.numel()))
                    continue

            weight_copy = _get_layer_channel_importance(m, name, require_a_row_norm=True)
            prune_score = _make_prune_score(weight_copy, m, name)
            mask = prune_score.gt(thre).float().to(device)
            if _use_fixed_dsf_mask(name):
                mask = channel_mask_dict[name].to(device).float()

                if mask.numel() != weight_copy.numel():
                    msg = (
                        f"mask size mismatch for {name}: "
                        f"mask={mask.numel()} vs out_channels={weight_copy.numel()}"
                    )
                    if bool(getattr(args, "strict_a_row_norm", True)):
                        raise RuntimeError(f"{msg}. Cannot use an invalid A-row-norm hard mask.")
                    print(f"[WARN] {msg}, fallback to score threshold.")
                    mask = prune_score.gt(thre).float().to(device)

            if args.model in ['resnet56', 'resnet110']:
                # 更强保护前两个 block 的 conv_bn1，
                # 同时把普通层压低一点，尽量不损失总 FLOPs
                layerwise_keep = None
                layerwise_bounds_only = False
                if _use_resnet56_adaptive_protect(args):
                    layerwise_keep = None
                    layerwise_bounds_only = False
                elif _use_resnet56_layerwise_prune(args):
                    layerwise_keep = _resnet56_layerwise_keep_ratio(name, args)
                    layerwise_bounds_only = False
                elif _use_resnet56_layerwise_bounds_only(args):
                    layerwise_keep = _resnet56_layerwise_keep_ratio(name, args)
                    layerwise_bounds_only = layerwise_keep is not None
                elif _use_resnet110_layerwise_prune(args):
                    layerwise_keep = _resnet110_layerwise_keep_ratio(name, args)
                    layerwise_bounds_only = _use_resnet110_layerwise_bounds_only(args)
                elif _use_resnet110_layerwise_bounds_only(args):
                    layerwise_keep = _resnet110_layerwise_keep_ratio(name, args)
                    layerwise_bounds_only = True

                if _use_resnet56_adaptive_protect(args):
                    base_floor = _resnet56_base_floor_ratio(name, args)
                    if base_floor is not None:
                        parsed = _parse_cifar_basicblock_name(name)
                        stage_mean = None
                        if parsed is not None:
                            stage_mean = resnet56_stage_score_mean.get(parsed[0])
                        score_mean = float(weight_copy.detach().float().mean().item())
                        rel = 1.0
                        if stage_mean is not None and stage_mean > 1e-12:
                            rel = score_mean / stage_mean
                        rel = min(1.5, max(0.5, rel))
                        span = float(getattr(args, "resnet56_adaptive_span", 0.10))
                        min_keep_ratio = base_floor + span * (rel - 1.0)
                        limits = _resnet56_adaptive_limits(name, args)
                        max_keep_ratio = float(getattr(args, "resnet56_adaptive_max_keep", 1.0))
                        if limits is not None:
                            min_override, max_keep_ratio = limits
                            if min_override is not None:
                                min_keep_ratio = max(min_keep_ratio, min_override)
                        min_keep_ratio = min(1.0, max(0.01, min_keep_ratio))
                        min_keep = max(2, int(np.ceil(mask.shape[0] * min_keep_ratio)))
                    elif conv_local in sensitive_layer:
                        min_keep_ratio = float(getattr(args, "min_keep_sensitive", 0.50))
                        min_keep = max(4, int(mask.shape[0] * min_keep_ratio))
                        max_keep_ratio = float(getattr(args, "max_keep_sensitive", 1.0))
                    elif conv_local in medium_sensitive_layer:
                        min_keep_ratio = float(getattr(args, "min_keep_medium", 0.35))
                        min_keep = max(4, int(mask.shape[0] * min_keep_ratio))
                        max_keep_ratio = float(getattr(args, "max_keep_medium", 1.0))
                    else:
                        min_keep_ratio = float(getattr(args, "min_keep_normal", 0.15))
                        min_keep = max(2, int(mask.shape[0] * min_keep_ratio))
                        max_keep_ratio = float(getattr(args, "max_keep_normal", 1.0))
                elif layerwise_keep is not None:
                    if layerwise_bounds_only:
                        if _use_resnet56_layerwise_bounds_only(args):
                            min_keep_ratio, max_keep_ratio = _resnet56_layerwise_keep_bounds(name, args)
                        else:
                            min_keep_ratio, max_keep_ratio = _resnet110_layerwise_keep_bounds(name, args)
                        min_keep = max(2, int(np.ceil(mask.shape[0] * min_keep_ratio)))
                    else:
                        min_keep_ratio = layerwise_keep
                        max_keep_ratio = layerwise_keep
                        min_keep = max(2, int(round(mask.shape[0] * min_keep_ratio)))
                elif conv_local in sensitive_layer:
                    min_keep_ratio = float(getattr(args, "min_keep_sensitive", 0.50))
                    min_keep = max(4, int(mask.shape[0] * min_keep_ratio))
                    max_keep_ratio = float(getattr(args, "max_keep_sensitive", 1.0))
                elif conv_local in medium_sensitive_layer:
                    min_keep_ratio = float(getattr(args, "min_keep_medium", 0.35))
                    min_keep = max(4, int(mask.shape[0] * min_keep_ratio))
                    max_keep_ratio = float(getattr(args, "max_keep_medium", 1.0))
                else:
                    min_keep_ratio = float(getattr(args, "min_keep_normal", 0.15))
                    min_keep = max(2, int(mask.shape[0] * min_keep_ratio))
                    max_keep_ratio = float(getattr(args, "max_keep_normal", 1.0))

                if max_keep_ratio >= 1.0:
                    max_keep = int(mask.shape[0])
                else:
                    if layerwise_keep is not None and not layerwise_bounds_only:
                        max_keep = int(round(mask.shape[0] * max_keep_ratio))
                    else:
                        max_keep = int(mask.shape[0] * max_keep_ratio)
                    max_keep = max(min_keep, max_keep)
                    max_keep = min(int(mask.shape[0]), max_keep)

                mask, remain = _repair_mask_with_keep_bounds(weight_copy, mask, min_keep, max_keep)

            elif args.model == 'resnet50':
                remain = int(torch.sum(mask).item())

                layerwise_keep = None
                layerwise_bounds = None
                if _use_resnet50_layerwise_prune(args):
                    layerwise_bounds = _resnet50_layerwise_keep_bounds(name, args)
                    if layerwise_bounds is None:
                        layerwise_keep = _resnet50_layerwise_keep_ratio(name, args)

                if layerwise_bounds is not None:
                    min_keep_ratio, max_keep_ratio = layerwise_bounds
                    repair_tag = 'resnet50 layerwise repair!'
                elif layerwise_keep is not None:
                    min_keep_ratio = layerwise_keep
                    max_keep_ratio = layerwise_keep
                    repair_tag = 'resnet50 layerwise repair!'
                elif conv_local in sensitive_layer:
                    min_keep_ratio = float(getattr(args, "min_keep_sensitive", args.adjust))
                    max_keep_ratio = float(getattr(args, "max_keep_sensitive", 1.0))
                    repair_tag = 'sensitive layer repair!'
                elif conv_local in medium_sensitive_layer or conv_local in sensitive_layer1:
                    min_keep_ratio = float(getattr(args, "min_keep_medium", args.adjust + 0.1))
                    max_keep_ratio = float(getattr(args, "max_keep_medium", 1.0))
                    repair_tag = 'medium sensitive layer repair!'
                else:
                    min_keep_ratio = float(getattr(args, "min_keep_normal", 0.3))
                    max_keep_ratio = float(getattr(args, "max_keep_normal", 1.0))
                    repair_tag = 'repair!'

                min_keep = max(1, int(mask.shape[0] * min_keep_ratio))
                min_keep = min(int(mask.shape[0]), min_keep)
                if max_keep_ratio >= 1.0:
                    max_keep = int(mask.shape[0])
                else:
                    max_keep = max(min_keep, int(mask.shape[0] * max_keep_ratio))
                    max_keep = min(int(mask.shape[0]), max_keep)

                mask, remain_after_repair = _repair_mask_with_keep_bounds(weight_copy, mask, min_keep, max_keep)
                if remain_after_repair != remain:
                    print(repair_tag)
                remain = remain_after_repair
            else:
                remain = int(torch.sum(mask).item())

            # 用途：保存 hard mask 作用前后的权重，以及每层保留通道数
            layer_records.append(
                {
                    "name": name,
                    "mask": mask.clone(),
                    "score": weight_copy.detach().float().to(device),
                    "weight": m.weight.data.detach().clone(),
                    "is_prunable": True,
                    "is_main_cfg": True,
                    "min_keep": int(min_keep) if args.model in ['resnet56', 'resnet110', 'resnet50'] else 1,
                    "max_keep": int(max_keep) if args.model in ['resnet56', 'resnet110', 'resnet50'] else int(mask.numel()),
                }
            )

            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], remain))

            conv_local += 1

        elif isinstance(m, nn.MaxPool2d):
            if args.model == 'vgg16':
                cfg.append('M')

    _adjust_records_to_target_flops(layer_records)
    pruned = _sync_records_to_outputs(layer_records)
    final_drop, final_flops, final_base_flops = _estimate_flops_drop_from_records(layer_records)
    if final_drop is not None:
        print(
            f"[Target-FLOPs] final cfg FLOPs drop estimate: {final_drop:.4f}% "
            f"({final_flops:.0f}/{final_base_flops:.0f})"
        )
    if args.model == 'resnet50' and args.dataset == 'imagenet' and _is_main_process(args):
        base_profile = _resnet50_imagenet_profile_from_cfg(None)
        pruned_profile = _resnet50_imagenet_profile_from_cfg(cfg)
        if base_profile is not None and pruned_profile is not None:
            base_flops, base_params = base_profile
            pruned_flops, pruned_params = pruned_profile
            flops_drop = 100.0 - pruned_flops / max(base_flops, 1.0) * 100.0
            params_drop = 100.0 - pruned_params / max(base_params, 1.0) * 100.0
            speedup = base_flops / max(pruned_flops, 1.0)
            print(
                f"[Model-Stats] baseline_flops={base_flops / 1e9:.6f}G "
                f"baseline_params={base_params / 1e6:.6f}M"
            )
            print(
                f"[Model-Stats] pruned_flops={pruned_flops / 1e9:.6f}G "
                f"pruned_params={pruned_params / 1e6:.6f}M "
                f"flops_drop={flops_drop:.4f}% params_drop={params_drop:.4f}% "
                f"speedup={speedup:.4f}x"
            )
            summary_dir = os.path.join(getattr(args, "save_dir", "save_temp"), "config")
            os.makedirs(summary_dir, exist_ok=True)
            summary_path = os.path.join(summary_dir, "structured_pruning_summary.json")
            with open(summary_path, "w") as f:
                json.dump(
                    {
                        "model": args.model,
                        "dataset": args.dataset,
                        "cfg": [int(v) for v in cfg],
                        "cfg_length": len(cfg),
                        "baseline_flops": base_flops,
                        "baseline_params": base_params,
                        "pruned_flops": pruned_flops,
                        "pruned_params": pruned_params,
                        "flops_drop_percent": flops_drop,
                        "params_drop_percent": params_drop,
                        "speedup": speedup,
                    },
                    f,
                    indent=2,
                )
            print(f"[Model-Stats] saved summary to: {summary_path}")

    if _is_main_process(args):
        dataset_name = str(args.dataset).upper().replace("_", "-")
        model_name = str(args.model).replace("resnet", "ResNet-").replace("vgg", "VGG-")


    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    end_mask = cfg_mask[layer_id_in_cfg]
    print(cfg)


    if args.model == 'resnet56':
        sparse_model = resnet56_X(num_classes=num_classes, nums=cfg)
    elif args.model == 'resnet110':
        sparse_model = resnet110_X(num_classes=num_classes, nums=cfg)
    elif args.model == 'vgg16':
        sparse_model = vgg16_bn(cfg)
    elif args.model == 'resnet50':
        sparse_model = pruned_resnet50(cfg=cfg, dataset=args.dataset, num_classes=num_classes)
    if _is_main_process(args):
        if _is_distributed(args):
            print('[DDP] sparse model built; skip full model dump before DDP wrapping')
        else:
            print(sparse_model)

    conv_local = 1
    for [m0, m1] in zip(model.modules(), sparse_model.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]


        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(
                end_mask.cpu().numpy())))
            if conv_local in down_locate:
                print('downsample')
                idx0 = np.squeeze(np.argwhere(
                    np.asarray(torch.ones(m0.weight.shape[1]).cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(torch.ones(
                    m0.weight.shape[0]).cpu().numpy())))  # Preserve shortcut projection channels.
            conv_local += 1

            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[
                :, idx0.tolist(), :, :].clone()  # Slice input channels, then output channels.
            w1 = w1[idx1.tolist(), :, :, :].clone()
            if args.model == 'vgg16':
                w2 = m0.bias.data[idx1.tolist()].clone()
            m1.weight.data = w1.clone()
            if args.model == 'vgg16':
                m1.bias.data = w2.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    if _is_main_process(args):
        _save_pruned_cfg(args, cfg, cfg_mask, num_classes)
        init_ckpt_dir = os.path.join(getattr(args, "save_dir", "save_temp"), "finetune_model")
        os.makedirs(init_ckpt_dir, exist_ok=True)
        init_ckpt_path = os.path.join(init_ckpt_dir, f"{args.model}_structured_pruned_init.pth")
        torch.save(
            {
                "model_state_dict": sparse_model.state_dict(),
                "cfg": cfg,
                "cfg_mask": [mask.detach().cpu() for mask in cfg_mask],
                "model": args.model,
                "dataset": args.dataset,
                "num_classes": num_classes,
            },
            init_ckpt_path,
        )
        print(f"===> Saved structured-pruned init checkpoint to: {init_ckpt_path}")

    if _is_main_process(args) and not _is_distributed(args):
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            shape = (1, 3, 32, 32)
        elif args.dataset == 'imagenet':
            shape = (1, 3, 224, 224)
        input_data = torch.randn(shape)
        if args.model == 'resnet56':
            new_model = resnet56(num_classes=num_classes)
        elif args.model == 'resnet110':
            new_model = resnet110(num_classes=num_classes)
        elif args.model == 'vgg16':
            new_model = vgg16_bn()
        elif args.model == 'resnet50':
            new_model = pruned_resnet50(dataset=args.dataset, num_classes=num_classes)
        print('new model : ')
        print(new_model)
        print('sparse model : ', sparse_model)
        # Profile the dense model in evaluation mode.
        was_training = new_model.training
        new_model.eval()
        with torch.no_grad():
            flops1, params1 = profile(new_model.cpu(), (input_data,))
            print('sparse model flops before pruning : ', flops1)
            print('parameters number : ', params1)

            ori_params = print_model_param_nums(new_model.cpu())
            if args.dataset == 'cifar10' or args.dataset == 'cifar100':
                ori_flops = print_model_param_flops(new_model.cpu(), input_res=32, multiply_adds=False)
            elif args.dataset == 'imagenet':
                ori_flops = print_model_param_flops(new_model.cpu(), input_res=224, multiply_adds=False)
        if was_training:
            new_model.train()

        flops2, params2 = profile(sparse_model.cpu(), (input_data,))
        # Profile the compact model in evaluation mode.
        was_training = sparse_model.training
        sparse_model.eval()
        with torch.no_grad():
            sparse_params = print_model_param_nums(sparse_model.cpu())
            if args.dataset == 'cifar10' or args.dataset == 'cifar100':
                sparse_flops = print_model_param_flops(sparse_model.cpu(), input_res=32, multiply_adds=False)
            elif args.dataset == 'imagenet':
                sparse_flops = print_model_param_flops(sparse_model.cpu(), input_res=224, multiply_adds=False)
        if was_training:
            sparse_model.train()
        print('params : ', sparse_params)
        print('flops : ', sparse_flops)
        print('precent parameters drop: ', 100 - sparse_params / ori_params * 100)
        print('precent FLOPs drop : ', 100 - sparse_flops / ori_flops * 100)
        print('Speed up : ', ori_flops / sparse_flops)
    elif _is_main_process(args):
        print('[DDP] skip rank0-only THOP/FLOPs profiling before DDP wrapping to avoid collective timeout')
    return sparse_model
