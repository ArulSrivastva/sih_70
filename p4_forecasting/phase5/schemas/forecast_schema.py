"""The stable JSON contract a React/Leaflet frontend should consume.

Success response::

    {
      "status": "success",
      "model": {"experiment_id": "EXP005", "family": "GRU", "loss": "Huber"},
      "input": {"history_hours": 24, "history_steps": 5, "feature_count": 16},
      "forecast": [
        {"hours": 6,  "latitude": .., "longitude": .., "wind_speed_kmh": ..},
        {"hours": 12, "latitude": .., "longitude": .., "wind_speed_kmh": ..},
        {"hours": 24, "latitude": .., "longitude": .., "wind_speed_kmh": ..}
      ]
    }

Error response::

    {"status": "error",
     "error": {"code": "nan_values", "message": "history contains NaN; ..."}}

Error codes: missing_input, wrong_shape, bad_input, insufficient_history,
nan_values, infinite_values, out_of_range, malformed_timestamp,
non_monotonic_timestamps, timestamps_not_6hourly, internal_error.

Input history (dict form):  ``{"timestamps": [...5 ascending], "history": [
    {"lat": .., "lon": .. (0..360), "wind_speed": .., "pressure": ..,
     "sst": .., "wind_u": .., "wind_v": ..}, ... x5 ]}``
ordered chronologically ascending (t-24h ... t).
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import HISTORY_HOURS, HISTORY_STEPS, FEATURE_NAMES_RAW, N_FEATURES
from ..inference.output_contract import validate_response

TOP_LEVEL_KEYS = ["status", "model", "input", "forecast"]
FORECAST_KEYS = ["hours", "latitude", "longitude", "wind_speed_kmh"]
INPUT_KEYS = ["history_hours", "history_steps", "feature_count"]
MODEL_KEYS = ["experiment_id", "family", "loss"]

CANONICAL_INPUT_FIELDS = FEATURE_NAMES_RAW
INPUT_HORIZONS = [6, 12, 24]


def validate_service_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Frontend-schema validation; returns {"pass": bool, "problems": [..]}."""
    if not isinstance(response, dict):
        return {"pass": False, "problems": ["response is not a dict"]}
    report = validate_response(response)
    if response.get("status") == "error":
        if "error" not in response or "code" not in response.get("error", {}):
            report["problems"].append("error response missing 'error.code'")
    return {"pass": report.get("pass") and not report.get("problems"),
            "problems": report.get("problems", [])}


def describe_input_schema() -> Dict[str, Any]:
    return {
        "history_steps": HISTORY_STEPS,
        "history_hours": HISTORY_HOURS,
        "field_order": list(FEATURE_NAMES_RAW),
        "longitude_convention": "degrees east in [0, 360)",
        "cadence_hours": 6,
        "ordering": "chronologically ascending (oldest first)",
    }


def describe_output_schema() -> Dict[str, Any]:
    return {
        "top_level_keys": TOP_LEVEL_KEYS,
        "forecast_keys": FORECAST_KEYS,
        "horizons_hours": INPUT_HORIZONS,
        "input_block_keys": INPUT_KEYS,
        "model_block_keys": MODEL_KEYS,
        "feature_count": N_FEATURES,
    }