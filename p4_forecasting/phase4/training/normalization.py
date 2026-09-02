"""P4 Phase-4 normalization (train-only, deterministic).

Statistics are computed from the TRAINING split only.  Validation/test data
never influences the statistics.  Directional features are documented
explicitly; zero-variance features are handled deterministically (scale = 1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import FEATURE_NAMES, TARGET_NAMES

DIRECTIONAL_FEATURES = {
    "movement_direction": {
        "representation": "degrees clockwise from true north in [0,360); "
                          "0 at first step (zero-fill policy).",
        "normalization": "ordinary linear z-score with TRAIN-only statistics.",
        "note": "The locked 16-column contract prevents a sine/cosine embedding; "
                "the 0/360 wrap discontinuity is a documented limitation of "
                "linear z-scoring, not silently hidden.",
    },
    "environmental_wind_direction": {
        "representation": "meteorological FROM-direction, degrees clockwise "
                          "from true north in [0,360).",
        "normalization": "ordinary linear z-score with TRAIN-only statistics.",
        "note": "Same documented wrap limitation as movement_direction.",
    },
}


def _validate(arr: np.ndarray, name: str, tail: tuple) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != len(tail) + 1 or arr.shape[1:] != tail:
        raise ValueError(f"{name}: expected shape (N,{tail[0]},{tail[1]}); got {arr.shape}")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(f"{name}: NaN/Inf found; refusing to normalize invalid data")
    return arr


def _mean_std(arr: np.ndarray) -> tuple:
    return arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))


def compute_normalization_stats(X_train: np.ndarray, Y_train: np.ndarray) -> Dict[str, object]:
    """Train-only feature(16)/target(3) mean & std.

    Zero-variance features are NOT an error: std <= 0 is replaced by scale 1.0
    (identity) deterministically, and every such feature is named in the stats.
    """
    X = _validate(X_train, "X_train", (5, 16))
    Y = _validate(Y_train, "Y_train", (3, 3))

    f_mean, f_std = _mean_std(X)
    t_mean, t_std = _mean_std(Y)

    zero_std_feats = [
        FEATURE_NAMES[i] for i, s in enumerate(f_std) if not np.isfinite(s) or s <= 0.0]
    zero_std_targets = [
        TARGET_NAMES[i] for i, s in enumerate(t_std) if not np.isfinite(s) or s <= 0.0]

    f_std = np.where(np.isfinite(f_std) & (f_std > 0.0), f_std, 1.0)
    t_std = np.where(np.isfinite(t_std) & (t_std > 0.0), t_std, 1.0)

    return {
        "feature_mean": [float(v) for v in f_mean],
        "feature_std": [float(v) for v in f_std],
        "target_mean": [float(v) for v in t_mean],
        "target_std": [float(v) for v in t_std],
        "feature_order": list(FEATURE_NAMES),
        "target_order": list(TARGET_NAMES),
        "computed_from": {"split": "train", "policy": "TRAIN only; never val/test/combined"},
        "zero_std_features": zero_std_feats,
        "zero_std_targets": zero_std_targets,
        "zero_std_handling": "std<=0 replaced by scale 1.0 (identity), deterministically",
        "directional_features": DIRECTIONAL_FEATURES,
        "n_train_samples": int(X.shape[0]),
        "stats_schema": "phase4 v1",
    }


class Normalizer:
    """Z-score normalizer fitted on Phase-4 train-only statistics."""

    def __init__(self, stats: Dict[str, object]) -> None:
        required = {"feature_mean", "feature_std", "target_mean", "target_std"}
        missing = required - set(stats)
        if missing:
            raise ValueError(f"normalization stats missing keys: {sorted(missing)}")
        if list(stats.get("feature_order", FEATURE_NAMES)) != FEATURE_NAMES:
            raise ValueError("stats feature_order does not match the 16-feature contract")
        self.stats = stats
        self.feature_mean = np.asarray(stats["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(stats["feature_std"], dtype=np.float32)
        self.target_mean = np.asarray(stats["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(stats["target_std"], dtype=np.float32)

    @classmethod
    def from_path(cls, path: str | Path) -> "Normalizer":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def normalize_X(self, x):
        return (np.asarray(x, dtype=np.float32) - self.feature_mean) / self.feature_std

    def normalize_Y(self, y):
        return (np.asarray(y, dtype=np.float32) - self.target_mean) / self.target_std

    def denormalize_Y(self, yhat):
        return np.asarray(yhat, dtype=np.float32) * self.target_std + self.target_mean

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.stats, indent=2, ensure_ascii=False), encoding="utf-8")