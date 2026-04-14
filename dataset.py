from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

# ── Constants ────────────────────────────────────────────────────────────────
# IMG_SIZE=224 works with efficientnet_b0 and mobilenetv3.
# Switch to 380 only if using efficientnet_b4 (much slower, ~2× memory).
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

SEED = 42


# ── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(img_size: int = IMG_SIZE):
    """Returns (train_transform, val_transform) torchvision Compose pipelines."""
    train_tfm = T.Compose([
        T.Resize((img_size + 20, img_size + 20)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.RandomApply([T.RandomRotation(15)], p=0.3),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])

    val_tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    return train_tfm, val_tfm


# ── Dataset ──────────────────────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    """Loads real/fake face images from a DataFrame with path & label columns."""

    def __init__(self, df: pd.DataFrame, transform=None, img_size: int = IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
        except Exception:
            # Fallback for corrupted images
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, int(row["label"])


# ── DataLoaders ──────────────────────────────────────────────────────────────
def setup_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = IMG_SIZE,
):
    """
    Splits df into train / val / test and returns DataLoaders.

    Improvements over previous version:
    - persistent_workers=True  → avoids worker re-spawn cost each epoch
    - prefetch_factor=2        → pre-loads next batch while GPU trains
    - drop_last=True on train  → avoids tiny final batches that can hurt BatchNorm
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
    )

    train_tfm, val_tfm = get_transforms(img_size)

    train_ds = DeepfakeDataset(train_df, train_tfm, img_size)
    val_ds   = DeepfakeDataset(val_df,   val_tfm,   img_size)
    test_ds  = DeepfakeDataset(test_df,  val_tfm,   img_size)

    # Weighted sampler to handle class imbalance
    cc = train_df["label"].value_counts().sort_index().values
    cw = 1.0 / torch.tensor(cc, dtype=torch.float)
    sw = cw[train_df["label"].values]
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    # persistent_workers and prefetch_factor require num_workers > 0
    worker_kwargs = (
        dict(persistent_workers=True, prefetch_factor=2) if num_workers > 0 else {}
    )

    train_loader = DataLoader(
        train_ds, batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **worker_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **worker_kwargs,
    )

    return train_loader, val_loader, test_loader, test_df
