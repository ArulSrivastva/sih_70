"""Determinism tests: identical input -> identical output (CPU)."""

import numpy as np

from phase5.inference.predictor import configure_deterministic_cpu, load_predictor
from phase5.config import default_paths
from phase5.inference.preprocessing import engineer_history


def _hist():
    return np.array([
        [23.30, 68.50, 55.6, 994.0, 28.67, 8.5095, 1.2749],
        [23.40, 68.00, 64.8, 993.0, 28.26, 4.4456, 7.0864],
        [23.40, 67.10, 64.8, 993.0, 28.43, 7.9810, -3.3996],
        [23.60, 66.40, 64.8, 992.0, 28.69, 2.0987, -4.1294],
        [23.50, 65.70, 74.1, 991.0, 28.85, 2.4551, 4.3582],
    ], dtype=np.float32)


def test_repeated_forecast_identical():
    configure_deterministic_cpu()
    pred = load_predictor(default_paths())
    feats = engineer_history(_hist())
    a = pred.predict_features(feats)
    b = pred.predict_features(feats)
    assert np.array_equal(a, b)
    assert np.max(np.abs(a - b)) == 0.0


def test_determinism_tolerance_documented():
    pred = load_predictor(default_paths())
    feats = engineer_history(_hist())
    rep = pred.forecast_twice(feats)
    assert rep["pass"]
    assert rep["tolerance"] <= 1e-6


def test_cpu_only():
    import torch
    pred = load_predictor(default_paths())
    assert str(next(pred.model.parameters()).device) == "cpu"
    assert torch.get_num_threads() >= 1


def test_service_forecast_deterministic(service, example_history_dict):
    r1 = service.forecast(example_history_dict)
    r2 = service.forecast(example_history_dict)
    assert r1 == r2