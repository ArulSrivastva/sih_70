"""Phase-5 preprocessing = the exact Phase-4 preprocessing, reused read-only.

* feature engineering           -> phase4.features.feature_engineering
* normalize/denormalize stats    -> phase4.training.normalization (TRAIN only)

Feature order is re-read from ``normalization_stats.json`` (the audited Phase-4
contract) rather than hard-coded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import (DERIVED_FEATURES, FEATURE_NAMES, FEATURE_NAMES_RAW,
                      N_FEATURES, read_json)
from .input_validation import parse_history


def load_feature_order(stats_path: str | Path) -> List[str]:
    """Authoritative 16-column feature order from the Phase-4 contract."""
    stats = read_json(stats_path)
    order = [str(x) for x in stats["feature_order"]]
    if order != FEATURE_NAMES:
        raise ValueError(
            "normalization_stats feature_order does not match the locked "
            f"16-feature contract: {order}")
    return order


def load_normalizer(stats_path: str | Path):
    """Phase-4 train-only Normalizer (zero-variance handling included)."""
    from phase4.training.normalization import Normalizer  # lightweight import
    return Normalizer.from_path(stats_path)


def engineer_history(history7: np.ndarray) -> np.ndarray:
    """(5,7) raw history -> (5,16) engineered features (Phase-4 exact)."""
    from phase4.features.feature_engineering import features_from_history
    h = np.asarray(history7, dtype=np.float32)
    return features_from_history(h)


def normalize_features(features16: np.ndarray, normalizer) -> np.ndarray:
    """Z-score a (5,16) block with the train-only statistics."""
    x = np.asarray(features16, dtype=np.float32)
    if x.shape != (5, 16):
        raise ValueError(f"expected (5,16) feature block; got {x.shape}")
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("feature block contains NaN/Inf")
    return np.asarray(normalizer.normalize_X(x[None, ...]), dtype=np.float32)[0]


def denormalize_targets(pred_norm: np.ndarray, normalizer) -> np.ndarray:
    """(3,3) normalized targets -> physical [lat, lon, wind_speed_kmh]."""
    y = np.asarray(pred_norm, dtype=np.float32)
    if y.shape != (3, 3):
        raise ValueError(f"expected (3,3) target predictions; got {y.shape}")
    return np.asarray(normalizer.denormalize_Y(y), dtype=np.float32)


def check_first_step_zero_fill(features16: np.ndarray) -> bool:
    """True iff the 7 predecessor-dependent trend features are zero at t-24h."""
    x = np.asarray(features16, dtype=np.float32)
    return bool(np.all(x[0, 7:14] == 0.0))


def feature_contract_check(features16: np.ndarray, order: List[str]) -> Dict[str, object]:
    """Structural checks of the engineered block against the contract."""
    x = np.asarray(features16, dtype=np.float32)
    return {
        "shape": list(x.shape),
        "expected_shape": [5, N_FEATURES],
        "order": order,
        "order_matches_locked_contract": order == FEATURE_NAMES,
        "raw_seven_byte_identical_possible": bool(np.array_equal(
            x[:, :7], np.asarray(x, np.float32)[:, :7])),
        "first_step_zero_fill": check_first_step_zero_fill(x),
        "any_nan": bool(np.isnan(x).any()),
        "any_inf": bool(np.isinf(x).any()),
    }


def inspect_stats(stats_path: str | Path) -> Dict[str, object]:
    stats = read_json(stats_path)
    return {
        "computed_from": stats.get("computed_from"),
        "n_train_samples": stats.get("n_train_samples"),
        "feature_order": stats.get("feature_order"),
        "target_order": stats.get("target_order"),
        "zero_std_features": stats.get("zero_std_features"),
        "feature_mean_count": len(stats.get("feature_mean", [])),
        "feature_std_count": len(stats.get("feature_std", [])),
        "target_mean_count": len(stats.get("target_mean", [])),
        "target_std_count": len(stats.get("target_std", [])),
    }