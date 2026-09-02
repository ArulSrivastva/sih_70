"""Stable output contract for the forecasting service (frontend-friendly).

    { "status": "success",
      "model": {"experiment_id","family","loss"},
      "input": {"history_hours","history_steps","feature_count"},
      "forecast": [ {"hours","latitude","longitude","wind_speed_kmh"} x3 ] }
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

FORECAST_KEYS = ["hours", "latitude", "longitude", "wind_speed_kmh"]
HORIZON_HOURS = [6, 12, 24]


def build_forecast_list(pred: np.ndarray) -> List[Dict[str, float]]:
    """(3,3) denorm predictions -> list of {hours, latitude, longitude, wind}."""
    p = np.asarray(pred, dtype=np.float64)
    if p.shape != (3, 3):
        raise ValueError(f"expected (3,3) predictions; got {p.shape}")
    out = []
    for i, hours in enumerate(HORIZON_HOURS):
        out.append({
            "hours": int(hours),
            "latitude": float(p[i, 0]),
            "longitude": float(p[i, 1]),
            "wind_speed_kmh": float(p[i, 2]),
        })
    return out


def build_success_response(model_info: Dict[str, str],
                           forecast: List[Dict[str, float]],
                           history_hours: int = 24,
                           history_steps: int = 5,
                           feature_count: int = 16) -> Dict[str, Any]:
    return {
        "status": "success",
        "model": dict(model_info),
        "input": {
            "history_hours": int(history_hours),
            "history_steps": int(history_steps),
            "feature_count": int(feature_count),
        },
        "forecast": forecast,
    }


def build_error_response(code: str, message: str) -> Dict[str, Any]:
    return {"status": "error",
            "error": {"code": code, "message": message}}


def validate_forecast_list(forecast: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Structural + range validation of a forecast list; returns a report."""
    problems = []
    if not isinstance(forecast, list):
        return {"pass": False, "problems": ["forecast is not a list"]}
    if len(forecast) != 3:
        problems.append(f"expected 3 horizons; got {len(forecast)}")
    for i, f in enumerate(forecast):
        if not isinstance(f, dict):
            problems.append(f"forecast[{i}] not a dict")
            continue
        for key in FORECAST_KEYS:
            if key not in f:
                problems.append(f"forecast[{i}] missing key '{key}'")
        if f.get("hours") != HORIZON_HOURS[i]:
            problems.append(f"forecast[{i}].hours={f.get('hours')} "
                            f"!= expected {HORIZON_HOURS[i]}")
        lat, lon, wind = f.get("latitude"), f.get("longitude"), f.get("wind_speed_kmh")
        if isinstance(lat, (int, float)) and not (-90.0 <= lat <= 90.0):
            problems.append(f"forecast[{i}] latitude out of range: {lat}")
        if isinstance(lon, (int, float)) and not (0.0 <= lon < 360.0):
            problems.append(f"forecast[{i}] longitude not in [0,360): {lon}")
        if isinstance(wind, (int, float)) and wind < 0.0:
            problems.append(f"forecast[{i}] negative wind speed: {wind}")
    return {"pass": not problems, "problems": problems}


def validate_response(response: Dict[str, Any]) -> Dict[str, Any]:
    problems = []
    if response.get("status") != "success":
        return {"pass": False,
                "problems": ["status is not 'success'"]}
    for key in ("model", "input", "forecast"):
        if key not in response:
            problems.append(f"missing top-level key '{key}'")
    if "model" in response:
        for key in ("experiment_id", "family", "loss"):
            if key not in response["model"]:
                problems.append(f"model missing '{key}'")
    if "input" in response:
        for key in ("history_hours", "history_steps", "feature_count"):
            if key not in response["input"]:
                problems.append(f"input missing '{key}'")
    fc = validate_forecast_list(response.get("forecast", []))
    problems.extend(fc["problems"])
    return {"pass": not problems, "problems": problems}


def is_json_serializable(response: Any) -> bool:
    import json
    try:
        json.dumps(response)
        return True
    except (TypeError, ValueError):
        return False