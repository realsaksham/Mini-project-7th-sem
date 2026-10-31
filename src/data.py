"""
Data loading for PneumoniaMNIST (binary: 0 = normal, 1 = pneumonia).

PneumoniaMNIST ships with its own fixed train/val/test split (medmnist builds
these from the original chest X-ray dataset and resizes to 28x28 grayscale).
We use that split as-is rather than re-splitting, so results are comparable
to other work using this dataset.
"""

import random

import numpy as np
import torch
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def set_seed(seed: int) -> None:
    """
    Seed every RNG this project touches. Called once at the start of every
    entry point (train.py, ablation.py) before any data shuffling or weight
    initialization happens, so a full run is byte-for-byte reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Normalize into roughly [-1, 1]. PneumoniaMNIST images are single-channel
# grayscale, so a single mean/std pair is enough (no per-RGB-channel stats).
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def _make_subset(dataset, n: int, seed: int) -> Subset:
    """
    Deterministically pick `n` indices out of `dataset` using a seeded RNG.
    Using `np.random.default_rng(seed)` (rather than the global `np.random`
    state) means this selection doesn't depend on what other random calls
    happened before it -- the same `n`/`seed` always picks the same images.
    """
    rng = np.random.default_rng(seed)
    n = min(n, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)
    return Subset(dataset, indices.tolist())


def load_pneumoniamnist(cfg, subset: bool = False):
    """
    Build train/val/test DataLoaders for PneumoniaMNIST.

    Args:
        cfg: a configs.config.Config instance.
        subset: if True, cut each split down to cfg.subset_{train,val,test}_size
            images (fixed, seeded selection) for fast debugging.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = PneumoniaMNIST(split="train", transform=_TRANSFORM,
                               download=True, root=cfg.data_root)
    val_ds = PneumoniaMNIST(split="val", transform=_TRANSFORM,
                             download=True, root=cfg.data_root)
    test_ds = PneumoniaMNIST(split="test", transform=_TRANSFORM,
                              download=True, root=cfg.data_root)

    if subset:
        train_ds = _make_subset(train_ds, cfg.subset_train_size, cfg.seed)
        val_ds = _make_subset(val_ds, cfg.subset_val_size, cfg.seed)
        test_ds = _make_subset(test_ds, cfg.subset_test_size, cfg.seed)

    # A dedicated, seeded Generator for the train loader's shuffling means the
    # batch order is identical across runs even though training also consumes
    # randomness elsewhere (weight init, dropout if any, etc.).
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               generator=shuffle_gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
