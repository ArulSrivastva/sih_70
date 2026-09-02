"""Movement-vector baseline for cyclone forecasting (P4 Phase 2).

A simple, physically interpretable baseline:

  t-6h and t (the two most recent historical positions) define the current
  6-hourly movement vector of the storm:

      delta_lat = lat(t)   - lat(t-6h)
      delta_lon = lon(t)   - lon(t-6h)

  6-hour movement is estimated by this vector and extrapolated linearly:

      +6h  position = position(t) + 1 * delta
      +12h position = position(t) + 2 * delta
      +24h position = position(t) + 4 * delta

  Wind speed is forecast by persistence (future_wind = current_wind).

TARGET VALUES ARE NEVER USED.

Mathematical assumptions (documented):
  1. Constant velocity: direction and speed of motion do not change over the
     forecast window (a crude but common tropical-cyclone baseline).
  2. Linear displacement in lat/lon degrees (small-angle / short-window
     approximation; degrees are NOT treated as true distances here — track
     error is always measured in km using Haversine).
  3. Longitude wrap-around handled on a 0..360 domain: delta_lon is wrapped to
     (-180, 180] and the predicted longitude is wrapped back into (0, 360].
  4. Wind persistence: no attempt to model intensification or decay.
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
    """Produce a (3,3) movement-vector forecast from a (5,7) history block.

    Returns:
        forecast[0] = +6h, forecast[1] = +12h, forecast[2] = +24h,
        columns = [lat, lon, wind_speed]
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
        pred_lat = lat_t + mult * delta_lat
        pred_lon = _wrap_lon(lon_t + mult * delta_lon)
        rows.append([pred_lat, pred_lon, wind])
    return np.asarray(rows, dtype=np.float32)


class MovementVectorBaseline:
    """Callable movement-vector baseline."""

    def predict(self, history: Any) -> np.ndarray:
        return movement_vector_forecast(history)

    def __call__(self, history: Any) -> np.ndarray:
        return self.predict(history)