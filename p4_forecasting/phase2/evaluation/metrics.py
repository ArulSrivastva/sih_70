"""Forecast metrics for cyclone forecasting (P4 Phase 2).

For each horizon (+6h, +12h, +24h):
  * Track error (km, Haversine): mean, median, standard deviation
  * Wind: MAE and RMSE (km/h)

Forecasting only — no classification metrics here.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .geo_metrics import haversine_km

HORIZONS = ["6h", "12h", "24h"]
RESULT_HORIZON_INDEX = 1  # placeholder


def track_error_km(actual: np.ndarray, predicted: np.ndarray) -> List[np.ndarray]:
    """Per-horizon Haversine track error.

    Args:
        actual:    (N, 3, 3) true targets [lat, lon, wind_speed]
        predicted: (N, 3, 3) forecasts  [lat, lon, wind_speed]

    Returns:
        list of three 1-D arrays (km), indexed by horizon 6h, 12h, 24h.
    """
    actual = np.asarray(actual, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    if actual.shape != predicted.shape or actual.ndim != 3 or actual.shape[1:] != (3, 3):
        raise ValueError(f"expected (N,3,3) arrays; got {actual.shape} vs {predicted.shape}")

    errors = []
    for h in range(3):
        err = haversine_km(
            actual[:, h, 0], actual[:, h, 1],
            predicted[:, h, 0], predicted[:, h, 1],
        )
        errors.append(np.asarray(err, dtype=np.float64))
    return errors


def wind_mae(actual: np.ndarray, predicted: np.ndarray) -> List[float]:
    """Per-horizon wind MAE (km/h)."""
    actual = np.asarray(actual, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    return [float(np.mean(np.abs(actual[:, h, 2] - predicted[:, h, 2]))) for h in range(3)]


def wind_rmse(actual: np.ndarray, predicted: np.ndarray) -> List[float]:
    """Per-horizon wind RMSE (km/h)."""
    actual = np.asarray(actual, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    return [float(np.sqrt(np.mean((actual[:, h, 2] - predicted[:, h, 2]) ** 2))) for h in range(3)]


def summarize_split(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Full metric summary per horizon for one split.

    Returns dict keyed by horizon ("6h","12h","24h") with track error
    mean/median/std and wind MAE/RMSE.
    """
    actual = np.asarray(actual, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    track = track_error_km(actual, predicted)
    mae = wind_mae(actual, predicted)
    rmse = wind_rmse(actual, predicted)

    out: Dict[str, Dict[str, float]] = {}
    for h, name in enumerate(HORIZONS):
        out[name] = {
            "track_error_km_mean": float(np.mean(track[h])),
            "track_error_km_median": float(np.median(track[h])),
            "track_error_km_std": float(np.std(track[h])),
            "wind_mae": mae[h],
            "wind_rmse": rmse[h],
        }
    return out