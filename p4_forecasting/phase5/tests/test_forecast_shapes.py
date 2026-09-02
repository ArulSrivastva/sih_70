"""Forecast-shape tests: response structure and metadata fidelity."""

import numpy as np

from phase5.inference.predictor import load_predictor
from phase5.config import default_paths
from phase5.inference.preprocessing import engineer_history


def test_forecast_shapes(example_history_np):
    pred = load_predictor(default_paths())
    feats = engineer_history(example_history_np)
    out = pred.predict_features(feats)
    assert out.shape == (3, 3)


def test_forecast_values_are_physical(example_history_np):
    pred = load_predictor(default_paths())
    feats = engineer_history(example_history_np)
    out = pred.predict_features(feats)
    assert np.isfinite(out).all()
    assert np.all(out[:, 1] >= 0.0) and np.all(out[:, 1] < 360.0)
    assert np.all(np.abs(out[:, 0]) <= 90.0)
    assert np.all(out[:, 2] >= 0.0)


def test_response_container_shapes(service, example_history_dict):
    res = service.forecast(example_history_dict)
    assert len(res["forecast"]) == 3
    assert all(len(f) == 4 for f in res["forecast"])


def test_model_metadata_read_from_artifacts(service, example_history_dict):
    res = service.forecast(example_history_dict)
    assert res["model"]["experiment_id"] == "EXP005"
    assert res["model"]["family"] == "GRU"
    assert res["model"]["loss"] == "Huber"
    # cross-check against the audited champion file
    import json
    p = default_paths()
    champ = json.loads(p.champion_meta.read_text(encoding="utf-8"))
    assert res["model"]["experiment_id"] == champ["experiment_id"]


def test_compare_baselines_shape(service, example_history_dict):
    cmp = service.compare_baselines(example_history_dict)
    assert cmp["status"] == "success"
    for key in ("model_forecast", "persistence_forecast",
                "movement_vector_forecast"):
        assert len(cmp[key]) == 3
        for f in cmp[key]:
            assert set(f.keys()) == {"hours", "latitude", "longitude",
                                     "wind_speed_kmh"}