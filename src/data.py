from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, eval_tfms


def make_loaders(data_root, test_root=None, image_size=224, batch_size=32,
                 val_fraction=0.15, test_fraction=0.15,
                 num_workers=2, seed=42):
    """
    Development baseline.

    If test_root is supplied, it is used as the untouched holdout set.
    The train/validation split is still image-level and is NOT publication-grade
    patient-level validation. Replace it with patient-level/external validation
    before making research claims.
    """
    data_root = Path(data_root)
    train_tfms, eval_tfms = build_transforms(image_size)

    full = datasets.ImageFolder(data_root, transform=train_tfms)
    n = len(full)
    n_val = int(n * val_fraction)

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        full, [n - n_val, n_val], generator=generator
    )

    # Validation should not receive augmentation.
    eval_base = datasets.ImageFolder(data_root, transform=eval_tfms)
    val_ds.dataset = eval_base

    if test_root is not None:
        test_base = datasets.ImageFolder(test_root, transform=eval_tfms)
        test_ds = test_base
    else:
        # Development fallback only: create a random test split from the
        # training source. Do not use this for publication claims.
        n_test = int(len(train_ds) * test_fraction)
        train_n = len(train_ds) - n_test
        train_ds, test_ds = random_split(
            train_ds, [train_n, n_test], generator=generator
        )
        test_ds.dataset = eval_base

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, full.classes
