"""Model loading / IO contract / determinism-flags tests."""

import numpy as np

from phase5.config import HORIZON_HOURS, N_TARGETS, default_paths
from phase5.inference.predictor import load_predictor


def test_checkpoint_config_stats_exist():
    p = default_paths()
    assert p.champion_checkpoint.exists()
    assert p.champion_config.exists()
    assert p.normalization_stats.exists()


def test_predictor_loads():
    pred = load_predictor(default_paths())
    assert pred.feature_count == 16
    assert pred.history_steps == 5
    assert pred.horizon_hours == HORIZON_HOURS


def test_parameter_count_from_checkpoint_89577():
    pred = load_predictor(default_paths())
    assert pred.param_count == 89577


def test_model_input_output_shapes():
    import torch
    from phase4.models.gru import GRUCyclone
    from phase5.inference.predictor import load_predictor, configure_deterministic_cpu
    configure_deterministic_cpu()
    pred = load_predictor(default_paths())
    model = pred.model
    with torch.no_grad():
        o = model(torch.randn(1, 5, 16))
        o4 = model(torch.randn(4, 5, 16))
    assert tuple(o.shape) == (1, 3, 3)
    assert tuple(o4.shape) == (4, 3, 3)
    assert o.dtype == torch.float32


def test_config_contract_values():
    pred = load_predictor(default_paths())
    cfg = pred.config
    assert cfg["model"] == "gru"
    assert cfg["loss"] == "huber"
    assert cfg["hidden_size"] == 96
    assert cfg["layers"] == 2
    assert cfg["dropout"] == 0.1
    assert cfg["input_size"] == 16
    assert cfg["output_size"] == 9
    assert list(cfg["horizons_hours"]) == [6, 12, 24]


def test_targets_lat_lon_wind():
    p = default_paths()
    import json
    stats = json.loads(p.normalization_stats.read_text(encoding="utf-8"))
    assert stats["target_order"] == ["lat", "lon", "wind_speed"]