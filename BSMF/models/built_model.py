import torch
import torch.nn as nn


from models.hand_resnet50 import *
from models.resnet import ResNet_CIFAR, resnet56, resnet110


# VGG16 architecture for CIFAR inputs.
from collections import OrderedDict


VGG16_CIFAR_LAYOUT = (
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512,
)


def _expand_vgg16_channel_cfg(cfg):
    if cfg is None:
        return list(VGG16_CIFAR_LAYOUT)

    cfg = list(cfg)
    if any(value == "M" for value in cfg):
        if len(cfg) != len(VGG16_CIFAR_LAYOUT):
            raise ValueError("A full VGG16 cfg must include all pool markers")
        return cfg

    pool_positions = {
        index for index, value in enumerate(VGG16_CIFAR_LAYOUT) if value == "M"
    }
    expected_channels = len(VGG16_CIFAR_LAYOUT) - len(pool_positions)
    if len(cfg) != expected_channels:
        raise ValueError(
            f"VGG16 channel cfg must contain {expected_channels} widths, got {len(cfg)}"
        )

    channel_iter = iter(cfg)
    return [
        "M" if index in pool_positions else int(next(channel_iter))
        for index in range(len(VGG16_CIFAR_LAYOUT))
    ]


class VGG16CIFAR(nn.Module):
    """HRank VGG16-BN architecture for CIFAR inputs."""

    def __init__(self, cfg=None, num_classes=10, classifier_hidden=512):
        super().__init__()
        layers = OrderedDict()
        in_channels = 3
        for index, value in enumerate(_expand_vgg16_channel_cfg(cfg)):
            if value == "M":
                layers[f"pool{index}"] = nn.MaxPool2d(kernel_size=2, stride=2)
                continue
            out_channels = int(value)
            layers[f"conv{index}"] = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            )
            layers[f"norm{index}"] = nn.BatchNorm2d(out_channels)
            layers[f"relu{index}"] = nn.ReLU(inplace=True)
            in_channels = out_channels

        self.features = nn.Sequential(layers)
        classifier_hidden = int(classifier_hidden)
        if classifier_hidden < 1:
            raise ValueError(f"classifier_hidden must be positive, got {classifier_hidden}")
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(in_channels, classifier_hidden)),
                    ("norm1", nn.BatchNorm1d(classifier_hidden)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(classifier_hidden, int(num_classes))),
                ]
            )
        )

    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = nn.functional.avg_pool2d(outputs, kernel_size=2)
        outputs = outputs.view(outputs.size(0), -1)
        return self.classifier(outputs)


def vgg16_bn(cfg=None, num_classes=10, classifier_hidden=512):
    return VGG16CIFAR(
        cfg=cfg, num_classes=num_classes, classifier_hidden=classifier_hidden
    )


def model_factory(config: str, dataset: str, disable_auto_patch: bool = False) -> nn.Module:
    """Build one of the four experiment architectures."""
    classes = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200, "imagenet": 1000}
    dataset = dataset.lower()
    num_classes = classes[dataset]
    if config == "resnet56":
        return resnet56(num_classes)
    if config == "resnet110":
        return resnet110(num_classes)
    if config == "resnet50":
        return pruned_resnet50(pretrained=False, cfg=None, dataset=dataset, num_classes=num_classes)
    if config == "vgg16" and dataset in ("cifar10", "cifar100"):
        return vgg16_bn(num_classes=num_classes)
    raise ValueError(f"Unsupported experiment: {config}/{dataset}")
