"""P4 Phase-4 torch Dataset / DataLoader for the 16-feature canonical datasets.

Reads the Phase-4 feature dataset NPZ (results/feature_dataset/*.npz) READ-ONLY
and returns normalised ``(5,16)`` history / ``(3,3)`` target tensors plus
provenance metadata.  The raw arrays are never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..training.normalization import Normalizer


class ForecastingDataset(Dataset):
    """Normalised Phase-4 forecasting dataset (16-feature contract)."""

    def __init__(
        self,
        npz_path: str | Path,
        normalizer: Normalizer | None = None,
        metadata_path: str | Path | None = None,
        normalize_targets: bool = True,
    ) -> None:
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"source NPZ missing: {self.npz_path}")
        loaded = np.load(self.npz_path, allow_pickle=True)
        self.X = np.asarray(loaded["X"], dtype=np.float32)
        self.Y = np.asarray(loaded["Y"], dtype=np.float32)
        if self.X.ndim != 3 or self.X.shape[1:] != (5, 16):
            raise ValueError(f"X must be (N,5,16); got {self.X.shape}")
        if self.Y.ndim != 3 or self.Y.shape[1:] != (3, 3):
            raise ValueError(f"Y must be (N,3,3); got {self.Y.shape}")
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y row counts differ")

        self.normalizer = normalizer
        self.normalize_targets = normalize_targets

        self.metadata: pd.DataFrame | None = None
        if metadata_path is not None and Path(metadata_path).exists():
            self.metadata = pd.read_csv(metadata_path)
            if self.metadata.shape[0] != self.X.shape[0]:
                raise ValueError("metadata rows != X rows")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        if self.normalizer is not None:
            x = np.asarray(self.normalizer.normalize_X(x), dtype=np.float32)
            if self.normalize_targets:
                y = np.asarray(self.normalizer.normalize_Y(y), dtype=np.float32)
        meta: Dict[str, object] = {}
        if self.metadata is not None:
            row = self.metadata.iloc[int(idx)]
            meta = {
                "cyclone_id": str(row["cyclone_id"]),
                "t_zero": str(row["t_zero"]),
                "source_split": str(row["source_split"]),
                "original_sample_index": int(row["original_sample_index"]),
            }
        return {"history": torch.from_numpy(x), "target": torch.from_numpy(y),
                "metadata": meta}


def collate_forecasting(batch: list) -> Dict[str, Any]:
    histories = torch.stack([item["history"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    metadata = [item["metadata"] for item in batch]
    return {"history": histories, "target": targets, "metadata": metadata}