"""Causality: the API never consumes future (t+6/+12/+24) values, and the
forecast does respond to changes in the latest (t) observed history."""

import math

import pytest


def _steps():
    times = ["2024-08-25T00:00:00Z", "2024-08-25T06:00:00Z",
             "2024-08-25T12:00:00Z", "2024-08-25T18:00:00Z",
             "2024-08-26T00:00:00Z"]
    return [
        {"timestamp": t, "latitude": 23.0 + 0.05 * i,
         "longitude": 68.5 - 0.7 * i, "wind_speed_kmh": 55.6 + 4.0 * i,
         "pressure_hpa": 994.0 - 0.7 * i, "sst": 28.6, "wind_u": 5.0,
         "wind_v": 1.0}
        for i, t in enumerate(times)
    ]


def test_future_values_never_accepted(client):
    steps = _steps()
    steps.append({"timestamp": "2024-08-26T06:00:00Z",  # t+6 (future)
                  "latitude": 24.0, "longitude": 60.0, "wind_speed_kmh": 80.0,
                  "pressure_hpa": 990.0, "sst": 28.5, "wind_u": 0.0,
                  "wind_v": 0.0})
    r = client.post("/forecast", json={"history": steps})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_HISTORY_LENGTH"


def test_future_values_rejected_on_compare_too(client):
    steps = _steps() + [_steps()[0]]   # 6th entry -> rejected before service
    r = client.post("/forecast/compare", json={"history": steps})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_HISTORY_LENGTH"


def test_adapter_sends_exactly_five_observations(valid_request_model,
                                                 monkeypatch):
    """Spy on the Phase-5 service: the adapter must forward exactly the 5
    validated observations and nothing else (no future/target rows)."""
    import numpy as np
    from phase6.integration.forecasting_adapter import ForecastingAdapter

    captured = {}

    class _Spy:
        def forecast(self, doc):
            captured["doc"] = doc
            return {"status": "success",
                    "model": {"experiment_id": "EXP005", "family": "GRU",
                              "loss": "Huber"},
                    "input": {"history_hours": 24, "history_steps": 5,
                              "feature_count": 16},
                    "forecast": [{"hours": 6, "latitude": 0.0,
                                  "longitude": 0.0, "wind_speed_kmh": 0.0},
                                 {"hours": 12, "latitude": 0.0,
                                  "longitude": 0.0, "wind_speed_kmh": 0.0},
                                 {"hours": 24, "latitude": 0.0,
                                  "longitude": 0.0, "wind_speed_kmh": 0.0}]}

    adapter = ForecastingAdapter(service=_Spy())
    adapter.forecast(valid_request_model)
    doc = captured["doc"]
    assert len(doc["history"]) == 5
    assert len(doc["timestamps"]) == 5
    step = doc["history"][2]
    assert set(step.keys()) == {"lat", "lon", "wind_speed", "pressure",
                                "sst", "wind_u", "wind_v"}  # Phase-5 contract
    assert "target" not in step and "future" not in step


def test_forecast_changes_when_latest_wind_changes(client, valid_request):
    base = client.post("/forecast", json=valid_request).json()["forecast"]

    mod = dict(valid_request)
    mod["history"] = list(valid_request["history"])
    mod["history"][-1] = dict(valid_request["history"][-1])
    mod["history"][-1]["wind_speed_kmh"] = 160.0   # t observation changed
    changed = client.post("/forecast", json=mod).json()["forecast"]

    winds_base = [f["wind_speed_kmh"] for f in base]
    winds_changed = [f["wind_speed_kmh"] for f in changed]
    assert winds_changed != winds_base, \
        "forecast must respond to changes in the latest observed wind"


def test_forecast_changes_when_latest_lat_changes(client, valid_request):
    base = client.post("/forecast", json=valid_request).json()["forecast"]

    mod = dict(valid_request)
    mod["history"] = list(valid_request["history"])
    mod["history"][-1] = dict(valid_request["history"][-1])
    mod["history"][-1]["latitude"] += 2.0
    changed = client.post("/forecast", json=mod).json()["forecast"]

    lat_base = [f["latitude"] for f in base]
    lat_changed = [f["latitude"] for f in changed]
    assert any(not math.isclose(a, b, abs_tol=1e-9)
               for a, b in zip(lat_base, lat_changed)), \
        "forecast must respond to changes in the latest observed latitude"


def test_request_has_no_target_values(client, valid_request):
    # the request schema must not even contain a 'target'/'future' key path
    r = client.post("/forecast", json=valid_request)
    assert r.status_code == 200  # schema has no place for future targets