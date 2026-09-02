"""Normalization for P4 Phase 3.

Training-only statistics. The source NPZ arrays are never modified.

Statistics computed:
    feature_mean / feature_std : shape (7,)  over all (N,5) history steps of TRAIN only
    target_mean  / target_std  : shape (3,)  over all (N,3) horizons of TRAIN only

Transform:
    z = (x - mean) / std

Zero/NaN standard deviation is a hard error: we never silently replace invalid
statistics, and NaN/Inf in the source data stops execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

FEATURES = ["lat", "lon", "wind_speed", "pressure", "sst", "wind_u", "wind_v"]
TARGETS = ["lat", "lon", "wind_speed"]


def _validate_array(arr: np.ndarray, name: str, expected_tail: tuple) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != len(expected_tail) + 1 or arr.shape[1:] != expected_tail:
        raise ValueError(f"{name}: expected shape (N,{expected_tail[0]},{expected_tail[1]}); got {arr.shape}")
    if np.isnan(arr).any():
        raise ValueError(f"{name}: NaN found - refusing to normalize/denormalize with invalid data")
    if np.isinf(arr).any():
        raise ValueError(f"{name}: Inf found - refusing to normalize/denormalize with invalid data")
    return arr


def compute_normalization_stats(
    X_train: np.ndarray,
    Y_train: np.ndarray,
) -> Dict[str, object]:
    """Compute mean/std for features (7) and targets (3) from TRAINING data only.

    Raises ValueError on NaN/Inf or zero variance (never silently patched).
    """
    X = _validate_array(X_train, "X_train", (5, 7))
    Y = _validate_array(Y_train, "Y_train", (3, 3))

    feature_mean = X.mean(axis=(0, 1))
    feature_std = X.std(axis=(0, 1))
    target_mean = Y.mean(axis=(0, 1))
    target_std = Y.std(axis=(0, 1))

    for feat, std in zip(FEATURES, feature_std):
        if np.isnan(feature_std).any():
            raise ValueError("feature_std contains NaN (unexpected)")
        if std <= 0.0:
            raise ValueError(f"feature '{feat}' has zero variance (std={std}); "
                             "refusing to emit degenerate statistics")

    for tgt, std in zip(TARGETS, target_std):
        if np.isnan(target_std).any():
            raise ValueError("target_std contains NaN (unexpected)")
        if std <= 0.0:
            raise ValueError(f"target '{tgt}' has zero variance (std={std}); "
                             "refusing to emit degenerate statistics")

    return {
        "feature_mean": [float(v) for v in feature_mean],
        "feature_std": [float(v) for v in feature_std],
        "target_mean": [float(v) for v in target_mean],
        "target_std": [float(v) for v in target_std],
        "feature_order": list(FEATURES),
        "target_order": list(TARGETS),
        "computed_from": {"split": "train", "policy": "training data ONLY"},
        "n_history_steps_used": int(X.shape[0]),
    }


class Normalizer:
    """Z-score normalizer fitted on training-only means/stds."""

    def __init__(self, stats: Dict[str, object]) -> None:
        required = {"feature_mean", "feature_std", "target_mean", "target_std"}
        missing = required - set(stats.keys())
        if missing:
            raise ValueError(f"normalization stats missing keys: {sorted(missing)}")
        self.stats = stats
        self.feature_mean = np.asarray(stats["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(stats["feature_std"], dtype=np.float32)
        self.target_mean = np.asarray(stats["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(stats["target_std"], dtype=np.float32)

    @classmethod
    def from_path(cls, path: str | Path) -> "Normalizer":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def normalize_X(self, x: np.ndarray | "torch.Tensor") -> np.ndarray:
        return (x - self.feature_mean) / self.feature_std

    def normalize_Y(self, y: np.ndarray | "torch.Tensor") -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def denormalize_Y(self, yhat: np.ndarray | "torch.Tensor") -> np.ndarray:
        return yhat * self.target_std + self.target_mean

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.stats, indent=2, ensure_ascii=False), encoding="utf-8")