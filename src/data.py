"""
Data loading. Two binary chest X-ray datasets are supported, picked by
cfg.dataset:

  * "tb" -- Kaggle's Tuberculosis (TB) Chest X-ray Database
    (https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset),
    0 = normal, 1 = tuberculosis. ~3500 normal / ~700 TB PNGs at 512x512.
    It ships with no train/val/test split, so we build a stratified, seeded
    one (cfg.val_frac / cfg.test_frac).

  * "pneumoniamnist" -- 0 = normal, 1 = pneumonia. medmnist provides a fixed
    train/val/test split at 28x28 grayscale, which we use as-is so results
    are comparable to other work using this dataset.

Both paths return the same thing -- (image tensor (1, H, W), label tensor (1,))
batches -- so nothing downstream (models/train/evaluate) cares which is used.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
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


# Normalize into roughly [-1, 1]. Both datasets are used as single-channel
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


# ---------------------------------------------------------------------------
# TB Chest X-ray (Kaggle)
# ---------------------------------------------------------------------------

# Subfolder name -> label, as laid out in the extracted Kaggle archive.
_TB_CLASS_DIRS = {"Normal": 0, "Tuberculosis": 1}
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


class _ArrayDataset(Dataset):
    """In-memory (N, H, W) uint8 images + (N,) labels, medmnist-style output."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])  # (H, W) uint8 -> (1, H, W) float
        # Shape (1,) to match medmnist, since train/evaluate call labels.squeeze(1).
        label = torch.tensor([self.labels[idx]], dtype=torch.long)
        return img, label


class _FeatureDataset(Dataset):
    """Precomputed (N, C, H, W) float feature maps + (N,) labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], torch.tensor([self.labels[idx]], dtype=torch.long)


def _load_tb_arrays(cfg):
    """
    Read every TB-dataset image, convert to grayscale, resize to
    cfg.image_size, and return (images uint8 (N, H, W), labels int64 (N,)).

    Decoding ~4200 512x512 PNGs takes a while, so the resized arrays are
    cached to data_root/tb_chest_xray_{size}.npz and reused on later runs.
    """
    cache_path = os.path.join(cfg.data_root, f"tb_chest_xray_{cfg.image_size}.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["images"], cached["labels"]

    if not os.path.isdir(cfg.tb_root):
        raise FileNotFoundError(
            f"TB dataset not found at {cfg.tb_root!r}. Download it from "
            "https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset "
            "and extract it so that folder contains the Normal/ and Tuberculosis/ "
            "subfolders (or point Config.tb_root at wherever you extracted it)."
        )

    paths, labels = [], []
    for class_dir, label in _TB_CLASS_DIRS.items():
        class_path = os.path.join(cfg.tb_root, class_dir)
        if not os.path.isdir(class_path):
            raise FileNotFoundError(f"Expected class folder {class_path!r} is missing.")
        for name in sorted(os.listdir(class_path)):
            if name.lower().endswith(_IMAGE_EXTS):
                paths.append(os.path.join(class_path, name))
                labels.append(label)

    print(f"[data] decoding {len(paths)} TB images to {cfg.image_size}x{cfg.image_size} "
          f"(one-time; cached to {cache_path})")
    images = np.empty((len(paths), cfg.image_size, cfg.image_size), dtype=np.uint8)
    for i, path in enumerate(paths):
        # Some images in the archive are stored as RGB; convert("L") makes
        # everything single-channel grayscale.
        with Image.open(path) as im:
            im = im.convert("L").resize((cfg.image_size, cfg.image_size), Image.BILINEAR)
            images[i] = np.asarray(im)
    labels = np.asarray(labels, dtype=np.int64)

    os.makedirs(cfg.data_root, exist_ok=True)
    np.savez_compressed(cache_path, images=images, labels=labels)
    return images, labels


def _load_tb_splits(cfg):
    images, labels = _load_tb_arrays(cfg)
    idx = np.arange(len(labels))

    # Stratified so the ~5:1 normal:TB ratio is preserved in every split;
    # seeded (cfg.split_seed) so the split is identical across runs, seeds,
    # models, and every ablation row.
    train_idx, temp_idx = train_test_split(
        idx, test_size=cfg.val_frac + cfg.test_frac,
        stratify=labels, random_state=cfg.split_seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=cfg.test_frac / (cfg.val_frac + cfg.test_frac),
        stratify=labels[temp_idx], random_state=cfg.split_seed,
    )

    if cfg.train_fraction < 1.0:
        # Low-data experiments: a stratified slice of the training split.
        # Seeded by cfg.seed, so each seed also sees a different slice.
        train_idx, _ = train_test_split(
            train_idx, train_size=cfg.train_fraction,
            stratify=labels[train_idx], random_state=cfg.seed,
        )

    if cfg.input_transform != "none":
        # Quantum (or control) feature maps, precomputed for every image.
        from src.quanvolution import cached_features
        feats = cached_features(cfg.input_transform, images, cfg)

        def make(split_idx):
            return _FeatureDataset(feats[split_idx], labels[split_idx])
    else:
        def make(split_idx):
            return _ArrayDataset(images[split_idx], labels[split_idx], _TRANSFORM)

    return make(train_idx), make(val_idx), make(test_idx)


# ---------------------------------------------------------------------------
# PneumoniaMNIST
# ---------------------------------------------------------------------------


def _load_pneumoniamnist_splits(cfg):
    # Imported lazily so the TB path doesn't require medmnist to be installed.
    from medmnist import PneumoniaMNIST

    return tuple(
        PneumoniaMNIST(split=split, transform=_TRANSFORM, download=True, root=cfg.data_root)
        for split in ("train", "val", "test")
    )


_SPLIT_LOADERS = {
    "tb": _load_tb_splits,
    "pneumoniamnist": _load_pneumoniamnist_splits,
}


def load_data(cfg, subset: bool = False):
    """
    Build train/val/test DataLoaders for cfg.dataset.

    Args:
        cfg: a configs.config.Config instance.
        subset: if True, cut each split down to cfg.subset_{train,val,test}_size
            images (fixed, seeded selection) for fast debugging.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = _SPLIT_LOADERS[cfg.dataset](cfg)

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
