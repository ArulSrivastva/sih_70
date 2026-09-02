"""PyTorch Dataset for cyclone forecasting (P4 Phase 2).

Clean pass-through of the canonical forecasting arrays:
  - loads NPZ (X, Y) + metadata CSV
  - preserves float32 dtype
  - preserves feature order  : [lat, lon, wind_speed, pressure, sst, wind_u, wind_v]
  - preserves target order   : [lat, lon, wind_speed]
  - does NOT normalize, impute, interpolate, or modify source NPZ files
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURES = ["lat", "lon", "wind_speed", "pressure", "sst", "wind_u", "wind_v"]
TARGETS = ["lat", "lon", "wind_speed"]
HISTORY_STEPS = ["t-24h", "t-18h", "t-12h", "t-6h", "t"]
HORIZONS = ["+6h", "+12h", "+24h"]


class CycloneForecastingDataset(Dataset):
    """Iterable dataset yielding raw history/target tensors + metadata."""

    def __init__(
        self,
        npz_path: str | Path,
        metadata_path: str | Path | None = None,
        validate: bool = False,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.metadata_path = None if metadata_path is None else Path(metadata_path)

        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ source missing: {self.npz_path}")
        if self.metadata_path is not None and not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata source missing: {self.metadata_path}")

        loaded = np.load(self.npz_path, allow_pickle=True)
        self.X = np.asarray(loaded["X"], dtype=np.float32)
        self.Y = np.asarray(loaded["Y"], dtype=np.float32)
        self.features = [str(x) for x in loaded["features"]]
        self.targets = [str(x) for x in loaded["targets"]]

        if self.X.ndim != 3 or self.X.shape[1:] != (5, 7):
            raise ValueError(f"X must be (N,5,7); got {self.X.shape}")
        if self.Y.ndim != 3 or self.Y.shape[1:] != (3, 3):
            raise ValueError(f"Y must be (N,3,3); got {self.Y.shape}")
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y row counts differ")

        self.metadata: Optional[pd.DataFrame] = None
        if self.metadata_path is not None:
            self.metadata = pd.read_csv(self.metadata_path)
            if self.metadata.shape[0] != self.X.shape[0]:
                raise ValueError(
                    f"metadata rows ({self.metadata.shape[0]}) != X rows ({self.X.shape[0]})")

        if validate:
            if int(np.isnan(self.X).sum() + np.isnan(self.Y).sum()):
                raise ValueError("NaN found in dataset")
            if int(np.isinf(self.X).sum() + np.isinf(self.Y).sum()):
                raise ValueError("Inf found in dataset")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        history = torch.from_numpy(np.ascontiguousarray(self.X[idx]))
        target = torch.from_numpy(np.ascontiguousarray(self.Y[idx]))

        meta: Dict[str, object] = {}
        if self.metadata is not None:
            row = self.metadata.iloc[int(idx)]
            meta = {
                "cyclone_id": str(row["cyclone_id"]),
                "t_zero": str(row["t_zero"]),
                "t_minus_24h": str(row["t_minus_24h"]),
                "t_plus_24h": str(row["t_plus_24h"]),
            }
            if "quality_status" in row.index:
                meta["quality_status"] = str(row["quality_status"])

        return {"history": history, "target": target, "metadata": meta}

    def cyclones(self) -> List[str]:
        if self.metadata is None:
            raise RuntimeError("cyclones() requires metadata_path")
        return sorted(self.metadata["cyclone_id"].unique().tolist())


def collate_forecasting(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of dataset items into a batched dict.

    history/target are stacked into tensors of shape (B,5,7) and (B,3,3).
    metadata is returned as a parallel Python list of dicts (documented
    alternative to support batching while preserving per-sample metadata).
    """
    histories = torch.stack([item["history"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    metadata = [item["metadata"] for item in batch]
    return {"history": histories, "target": targets, "metadata": metadata}