"""Persistence baseline (Phase-2 methodology, reproduced exactly).

For every horizon (+6h/+12h/+24h):
    lat  = history[-1, 0]
    lon  = history[-1, 1]
    wind = history[-1, 2]

No future information is used.  Input is never modified.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LAT_IDX, LON_IDX, WIND_IDX = 0, 1, 2


def persistence_forecast(history: np.ndarray) -> np.ndarray:
    """(5,7) history -> (3,3) persistence forecast [lat, lon, wind]."""
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 2 or history.shape != (5, 7):
        raise ValueError(f"history must be (5,7); got {history.shape}")
    lat = history[-1, LAT_IDX]
    lon = history[-1, LON_IDX]
    wind = history[-1, WIND_IDX]
    current = np.array([lat, lon, wind], dtype=np.float32)
    return np.tile(current, (3, 1))


class PersistenceBaseline:
    """Callable persistence baseline."""

    def predict(self, history: Any) -> np.ndarray:
        return persistence_forecast(history)

    def __call__(self, history: Any) -> np.ndarray:
        return self.predict(history)