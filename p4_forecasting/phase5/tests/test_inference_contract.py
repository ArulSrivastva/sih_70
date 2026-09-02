"""Inference contract tests: JSON contract, keys, coordinates, horizons."""

import json

import numpy as np

from phase5.inference.output_contract import (FORECAST_KEYS,
                                              build_forecast_list,
                                              is_json_serializable,
                                              validate_forecast_list,
                                              validate_response)
from phase5.schemas.forecast_schema import validate_service_response


def _expected_pred():
    return np.array([[22.0, 66.5, 80.0],
                     [22.5, 63.0, 85.0],
                     [23.0, 57.0, 95.0]], dtype=np.float32)


def test_forecast_list_keys_and_order():
    fc = build_forecast_list(_expected_pred())
    assert [f["hours"] for f in fc] == [6, 12, 24]
    for f in fc:
        assert list(f.keys()) == FORECAST_KEYS


def test_forecast_list_validates():
    fc = build_forecast_list(_expected_pred())
    report = validate_forecast_list(fc)
    assert report["pass"] and report["problems"] == []


def test_json_serializable():
    fc = build_forecast_list(_expected_pred())
    assert is_json_serializable(fc)
    json.dumps(fc)


def test_service_forecast_contract(service, example_history_dict):
    res = service.forecast(example_history_dict)
    assert res["status"] == "success"
    assert set(res.keys()) == {"status", "model", "input", "forecast"}
    assert res["model"]["experiment_id"] == "EXP005"
    assert res["input"]["history_hours"] == 24
    assert res["input"]["history_steps"] == 5
    assert res["input"]["feature_count"] == 16
    assert validate_service_response(res)["pass"]


def test_service_contract_forecast_horizons_and_values(service,
                                                       example_history_dict):
    res = service.forecast(example_history_dict)
    fc = res["forecast"]
    assert [f["hours"] for f in fc] == [6, 12, 24]
    for f in fc:
        assert -90.0 <= f["latitude"] <= 90.0
        assert 0.0 <= f["longitude"] < 360.0
        assert f["wind_speed_kmh"] >= 0.0
        for k in ("hours", "latitude", "longitude", "wind_speed_kmh"):
            assert isinstance(f[k], (int, float))


def test_nan_input_error_response(service):
    import numpy as np
    h = np.full((5, 7), np.nan, dtype=np.float32)
    res = service.forecast(h)
    assert res["status"] == "error"
    assert res["error"]["code"] == "nan_values"


def test_inf_input_error_response(service):
    import numpy as np
    h = np.zeros((5, 7), dtype=np.float32)
    h[2, 3] = np.inf
    res = service.forecast(h)
    assert res["status"] == "error"
    assert res["error"]["code"] == "infinite_values"


def test_error_response_matches_schema(service):
    res = service.forecast(np.zeros((3, 7), dtype=np.float32))
    assert res["status"] == "error"
    assert "error" in res and "code" in res["error"] and "message" in res["error"]


def test_response_contract_validation_passes(service, example_history_dict):
    res = service.forecast(example_history_dict)
    assert validate_response(res)["pass"]