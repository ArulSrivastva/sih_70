"""Strict input validation for the Phase-5 forecasting service.

Accepted input forms (all describing the SAME raw history):
  * numpy.ndarray of shape (5, 7) float32/float64  [lat, lon, wind_speed,
    pressure, sst, wind_u, wind_v] per HISTORY STEP, ordered chronologically
    ascending (t-24h ... t).
  * a dict:
        {"timestamps": [...5 ISO-8601 strings, ascending, 6-hourly...],
         "history":    [...5 dicts, each with the 7 canonical fields...]}
    ``timestamps`` is optional; when present every cadence / monotonicity /
    format rule is enforced.

Invalid data is rejected, never repaired.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Union

import numpy as np

from ..config import HISTORY_STEPS, RAW_FEATURES, SPACING_HOURS

HISTORY = "history"
STEPS = "steps"
TIMESTAMPS = "timestamps"

CANONICAL_FIELDS = ["lat", "lon", "wind_speed", "pressure", "sst",
                    "wind_u", "wind_v"]

# Physically-plausible closed intervals.  Longitude uses the canonical
# North-Indian-Ocean 0..360 convention used throughout the pipeline.
BOUNDS = {
    "lat": (-90.0, 90.0),
    "lon": (0.0, 360.0),
    "wind_speed": (0.0, 400.0),
    "pressure": (850.0, 1100.0),
    "sst": (-5.0, 45.0),
}


class InputError(ValueError):
    """Raised on invalid/incomplete input; never raised on valid input."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message}


def _require(cond: bool, code: str, message: str) -> None:
    if not cond:
        raise InputError(code, message)


def _contains_required_keys(step: Dict[str, Any]) -> bool:
    return all(k in step for k in CANONICAL_FIELDS)


def _parse_timestamp(value: Any) -> _dt.datetime:
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, str):
        try:
            return _dt.datetime.fromisoformat(value)
        except ValueError:
            raise InputError(
                "malformed_timestamp",
                f"cannot parse timestamp {value!r} (expected ISO-8601)")
    raise InputError(
        "malformed_timestamp",
        f"timestamp must be str/datetime; got {type(value).__name__}")


def _validate_range(x: np.ndarray, name: str, lo: float, hi: float) -> None:
    _require(bool(np.all((x >= lo) & (x <= hi))), "out_of_range",
             f"field '{name}' outside physical bound [{lo}, {hi}]")


def parse_history(history: Union[np.ndarray, List, Dict, None]) -> np.ndarray:
    """Validate + coerce to a (5, 7) float32 array; raise InputError otherwise."""
    if history is None:
        raise InputError("missing_input", "no history provided")

    if isinstance(history, np.ndarray):
        arr = history
        _require(arr.ndim == 2, "wrong_shape",
                 f"history array must be 2-D (5, 7); got {arr.ndim}-D")
        _require(arr.shape == (HISTORY_STEPS, RAW_FEATURES), "wrong_shape",
                 f"history must be ({HISTORY_STEPS}, {RAW_FEATURES}); "
                 f"got {tuple(arr.shape)}")
    elif isinstance(history, Dict):
        steps = history.get(HISTORY, history.get(STEPS))
        time_values = history.get(TIMESTAMPS)
        if not isinstance(steps, (list, tuple)) or not all(
                isinstance(s, dict) for s in steps):
            raise InputError("bad_input",
                             "dict input needs 'history': [5 step-dicts] "
                             "(or 'steps'); each with 7 canonical fields")
        _require(len(steps) == HISTORY_STEPS, "insufficient_history",
                 f"history must contain {HISTORY_STEPS} timesteps "
                 f"(24h of 6-hourly data); got {len(steps)}")
        for s in steps:
            _require(_contains_required_keys(s), "bad_input",
                     "each history step must contain the 7 canonical fields "
                     f"{CANONICAL_FIELDS}")
        arr = np.array([[float(s[k]) for k in CANONICAL_FIELDS]
                        for s in steps], dtype=np.float32)
        if time_values is not None:
            validate_timestamps(time_values)
    elif isinstance(history, (list, tuple)):
        _require(len(history) == HISTORY_STEPS, "insufficient_history",
                 f"history must contain {HISTORY_STEPS} timesteps; "
                 f"got {len(history)}")
        _require(all(isinstance(s, dict) and _contains_required_keys(s)
                     for s in history), "bad_input",
                 "each history step must contain the 7 canonical fields "
                 f"{CANONICAL_FIELDS}")
        arr = np.array([[float(s[k]) for k in CANONICAL_FIELDS]
                        for s in history], dtype=np.float32)
    else:
        raise InputError("bad_input",
                         f"unsupported history type {type(history).__name__}")

    _require(arr.shape == (HISTORY_STEPS, RAW_FEATURES), "wrong_shape",
             f"history must be ({HISTORY_STEPS}, {RAW_FEATURES}) in canonical "
             f"field order {CANONICAL_FIELDS}; got {tuple(arr.shape)}")
    _require(not np.isnan(arr).any(), "nan_values",
             "history contains NaN; refusing to forecast")
    _require(not np.isinf(arr).any(), "infinite_values",
             "history contains infinite values; refusing to forecast")

    for field in ("lat", "lon", "wind_speed", "pressure", "sst"):
        lo, hi = BOUNDS[field]
        idx = CANONICAL_FIELDS.index(field)
        _validate_range(arr[:, idx], field, lo, hi)

    return np.ascontiguousarray(arr, dtype=np.float32)


def validate_timestamps(time_values: List[Any]) -> List[_dt.datetime]:
    """Must be exactly 5 ISO-8601 timestamps, ascending, exactly 6h apart."""
    if not isinstance(time_values, (list, tuple)):
        raise InputError("bad_input", "'timestamps' must be a list")
    if len(time_values) != HISTORY_STEPS:
        raise InputError("insufficient_history",
                         f"expected {HISTORY_STEPS} timestamps (24h of "
                         f"6-hourly data); got {len(time_values)}")
    parsed = [_parse_timestamp(v) for v in time_values]
    for i in range(1, len(parsed)):
        gap = parsed[i] - parsed[i - 1]
        if gap <= _dt.timedelta(0):
            raise InputError("non_monotonic_timestamps",
                             "timestamps must be strictly increasing")
        if gap != _dt.timedelta(hours=SPACING_HOURS):
            raise InputError("timestamps_not_6hourly",
                             f"history cadence must be exactly "
                             f"{SPACING_HOURS} hours; found {gap}")
    return parsed