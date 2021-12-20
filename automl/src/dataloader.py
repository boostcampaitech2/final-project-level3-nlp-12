"""Tune Model.

- Author: Junghoon Kim, Jongkuk Lim, Jimyeong Kim
- Contact: placidus36@gmail.com, lim.jeikei@gmail.com, wlaud1001@snu.ac.kr
- Reference
    https://github.com/j-marple-dev/model_compression
"""
import glob
import os
from typing import Any, Dict, List, Tuple, Union

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder, VisionDataset

from src.utils.data import weights_for_balanced_classes
from src.utils.torch_utils import split_dataset_index


def create_dataloader(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Simple dataloader.

    Args:
        cfg: yaml file path or dictionary type of the data.

    Returns:
        train_loader
        valid_loader
        test_loader
    """
    # Data Setup
    train_dataset, val_dataset, test_dataset = get_dataset(
        data_path=config["DATA_PATH"],
        dataset_name=config["DATASET"],
        img_size=config["IMG_SIZE"],
        val_ratio=config["VAL_RATIO"],
        transform_train=config["AUG_TRAIN"],
        transform_test=config["AUG_TEST"],
        transform_train_params=config["AUG_TRAIN_PARAMS"],
        transform_test_params=config.get("AUG_TEST_PARAMS"),
    )

    return get_dataloader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config["BATCH_SIZE"],
    )


def get_dataset(
    data_path: str = "./save/data",
    dataset_name: str = "CIFAR10",
    img_size: float = 32,
    val_ratio: float=0.2,
    transform_train: str = "simple_augment_train",
    transform_test: str = "simple_augment_test",
    transform_train_params: Dict[str, int] = None,
    transform_test_params: Dict[str, int] = None,
) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    """Get dataset for training and testing."""
    if not transform_train_params:
        transform_train_params = dict()
    if not transform_test_params:
        transform_test_params = dict()

    # preprocessing policies
    transform_train = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        transform_train,
    )(dataset=dataset_name, img_size=img_size, **transform_train_params)
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        transform_test,
    )(dataset=dataset_name, img_size=img_size, **transform_test_params)

    label_weights = None
    # pytorch dataset
    if dataset_name == "TACO":
        train_path = os.path.join(data_path, "train")
        val_path = os.path.join(data_path, "val")
        test_path = os.path.join(data_path, "test")

        train_dataset = ImageFolder(root=train_path, transform=transform_train)
        val_dataset = ImageFolder(root=val_path, transform=transform_test)
        test_dataset = ImageFolder(root=test_path, transform=transform_test)

    else:
        Dataset = getattr(
            __import__("torchvision.datasets", fromlist=[""]), dataset_name
        )
        train_dataset = Dataset(
            root=data_path, train=True, download=True, transform=transform_train
        )
        # from train dataset, train: 80%, val: 20%
        train_length = int(len(train_dataset) * (1.0-val_ratio))
        train_dataset, val_dataset = random_split(
            train_dataset, [train_length, len(train_dataset) - train_length]
        )
        test_dataset = Dataset(
            root=data_path, train=False, download=False, transform=transform_test
        )
    return train_dataset, val_dataset, test_dataset


def get_dataloader(
    train_dataset: VisionDataset,
    val_dataset: VisionDataset,
    test_dataset: VisionDataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get dataloader for training and testing."""

    train_loader = DataLoader(
        dataset=train_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=batch_size,
        num_workers=10,
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=val_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=5
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=5
    )
    return train_loader, valid_loader, test_loader
