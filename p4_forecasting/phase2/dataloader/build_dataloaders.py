"""Build train/validation/test DataLoaders for P4 Phase 2.

Default batch size: 64, num_workers: 0 for portability. Validation and test
loaders are never shuffled. Training may be shuffled with a fixed seed for
reproducibility (the chronological split is already established by cyclone).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .forecasting_dataset import CycloneForecastingDataset, collate_forecasting

SPLITS = ["train", "val", "test"]
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 0
SEED = 42


def build_dataloaders(
    dataset_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_WORKERS,
    shuffle_train: bool = True,
    seed: int = SEED,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for all splits from a directory with {train,val,test}.npz.

    Returns {"train": loader, "val": loader, "test": loader}.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory missing: {dataset_dir}")

    loaders: Dict[str, DataLoader] = {}
    g = torch.Generator()
    g.manual_seed(seed)

    for split in SPLITS:
        npz = dataset_dir / f"{split}.npz"
        meta = dataset_dir / f"{split}_metadata.csv"
        if not npz.exists():
            raise FileNotFoundError(f"Missing dataset file: {npz}")
        dataset = CycloneForecastingDataset(npz, meta if meta.exists() else None)
        should_shuffle = split == "train" and shuffle_train
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_workers,
            collate_fn=collate_forecasting,
            generator=g if should_shuffle else None,
            pin_memory=False,
        )
    return loaders