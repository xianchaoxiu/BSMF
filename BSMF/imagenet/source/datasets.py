"""Data loaders for CIFAR and ImageNet image classification."""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CustomImageNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []

        classes = sorted(
            class_name
            for class_name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, class_name))
        )
        self.classes = classes
        self.class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        for i, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if not os.path.isfile(file_path):
                    continue
                self.file_paths.append(file_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image_path = self.file_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        label = np.array(label)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_cifar10(root):
    num_classes = 10
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root , train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root, train=False, transform=test_transform, download=False
    )

    return num_classes, train_dataset, test_dataset


def _build_imagenet_folder(train_dir: Path, val_dir: Path, train_transforms, val_transforms):
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"ImageNet train/val not found: {train_dir}, {val_dir}")

    train = datasets.ImageFolder(
        train_dir,
        transform=transforms.Compose(train_transforms),
    )
    val = datasets.ImageFolder(
        val_dir,
        transform=transforms.Compose(val_transforms),
    )

    if train.class_to_idx != val.class_to_idx:
        raise ValueError(
            f"ImageNet train/val class count mismatch: "
            f"train={len(train.classes)}, val={len(val.classes)}"
        )

    return len(train.classes), train, val


def get_imagenet(root: Path, standardize=True):

    num_classes = 1000

    # NB for ImageNet we don't have access to the test dataset and instead use
    # validation set as test.

    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
    ]

    val_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(),
    ]

    if standardize:
        normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_transforms += [normalizer]
        val_transforms += [normalizer]

    if (root / "train").exists() and (root / "val").exists():
        num_classes, train, val = _build_imagenet_folder(
            root / "train",
            root / "val",
            train_transforms,
            val_transforms,
        )
    elif (root / "ImageNet_Torchvision").exists():####不同的数据集格式
        train = CustomImageNetDataset(
            root / "ImageNet_Torchvision" / "train",
            transform=transforms.Compose(train_transforms),
        )
        val = datasets.ImageNet(
            root / "ImageNet_Torchvision",
            split="val",
            download=False,
            transform=transforms.Compose(val_transforms),
        )
        num_classes = len(train.classes)
    elif (root / "imagenet").exists():####不同的数据集格式
        num_classes, train, val = _build_imagenet_folder(
            root / "imagenet" / "train",
            root / "imagenet" / "val",
            train_transforms,
            val_transforms,
        )
    else:
        raise FileNotFoundError(
            f"ImageNet data not found under {root}. Expected either "
            f"{root}/train and {root}/val, or {root}/imagenet/train and {root}/imagenet/val."
        )

    return num_classes, train, val
