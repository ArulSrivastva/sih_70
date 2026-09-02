"""Movement-vector baseline (Phase-2 methodology, reproduced exactly).

Uses the two most recent positions (t-6h and t) as the 6-hourly movement
vector, then extrapolates linearly:

      +6h  position = position(t) + 1 * delta
      +12h position = position(t) + 2 * delta
      +24h position = position(t) + 4 * delta

Wind speed is forecast by persistence.  TARGET VALUES ARE NEVER USED.

Notes (kept from Phase 2): constant-velocity assumption; linear displacement
in lat/lon degrees (track error is still measured in km by Haversine);
longitude wrap handled on a 0..360 domain (delta wrapped into (-180, 180]).
"""

from __future__ import annotations

from typing import Any

import numpy as np

LAT_IDX, LON_IDX, WIND_IDX = 0, 1, 2
WINDOW = 6.0  # hours between the two latest history steps
MULTIPLIERS = {6: 1.0, 12: 2.0, 24: 4.0}  # hours ahead / 6h window


def _wrap_lon(lon: float) -> float:
    return float(lon % 360.0)


def movement_vector_forecast(history: np.ndarray) -> np.ndarray:
    """(5,7) history -> (3,3) movement-vector forecast [lat, lon, wind].

    Rows = +6h/+12h/+24h, columns = [lat, lon, wind_speed].
    """
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 2 or history.shape != (5, 7):
        raise ValueError(f"history must be (5,7); got {history.shape}")

    lat_t = float(history[-1, LAT_IDX])
    lon_t = float(history[-1, LON_IDX])
    lat_p6 = float(history[-2, LAT_IDX])
    lon_p6 = float(history[-2, LON_IDX])
    wind = float(history[-1, WIND_IDX])

    delta_lat = lat_t - lat_p6
    raw_dlon = lon_t - lon_p6
    delta_lon = (raw_dlon + 180.0) % 360.0 - 180.0  # wrap to (-180, 180]

    rows = []
    for hours in (6, 12, 24):
        mult = MULTIPLIERS[hours]
        rows.append([lat_t + mult * delta_lat,
                     _wrap_lon(lon_t + mult * delta_lon),
                     wind])
    return np.asarray(rows, dtype=np.float32)


class MovementVectorBaseline:
    """Callable movement-vector baseline."""

    def predict(self, history: Any) -> np.ndarray:
        return movement_vector_forecast(history)

    def __call__(self, history: Any) -> np.ndarray:
        return self.predict(history)