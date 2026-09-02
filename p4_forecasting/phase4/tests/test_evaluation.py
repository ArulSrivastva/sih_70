"""Unit tests for Phase-4 validation evaluation (TEST split is never used here)."""

from __future__ import annotations

import numpy as np
import pytest

from phase4.evaluation.evaluate import (
    evaluate_split,
    load_model_from_config,
    predict_split,
    write_json,
)
from phase4.models import ImprovedLSTM
from phase4.training.normalization import Normalizer
from phase4.training.train import TrainConfig, run_training
from tests.conftest import make_feature_npz, make_normalization_stats

pytest.importorskip("phase2.evaluation.metrics", reason="phase2 package unavailable")


@pytest.fixture()
def trained_eval(scratch):
    ds = make_feature_npz(scratch / "ds", n_train=40, n_val=10)
    stats = make_normalization_stats()
    ckpt = scratch / "ckpt.pt"
    hist = scratch / "history.json"
    out = run_training(
        dataset_dir=ds, stats=stats, checkpoint_path=ckpt, history_path=hist,
        cfg=TrainConfig(seed=42, learning_rate=1e-3, batch_size=8,
                        max_epochs=3, patience=2, device="cpu"),
        build_model=lambda: ImprovedLSTM(input_size=16, hidden_size=8,
                                         num_layers=1, output_size=9),
        build_criterion=lambda: np_to_torch_mse())
    yield ds, stats, ckpt


def np_to_torch_mse():
    import torch
    return torch.nn.MSELoss()


def _write_cfg(path, **over):
    cfg = {"model": "improved_lstm", "input_size": 16, "output_size": 9,
           "hidden_size": 8, "layers": 1, "dropout": 0.0}
    cfg.update(over)
    write_json(cfg, path)
    return path


def test_predict_split_shapes(trained_eval):
    ds, stats, ckpt = trained_eval
    cfg_path = _write_cfg(ds.parent / "cfg.json")
    model = load_model_from_config(cfg_path, ckpt)
    # predict_split expects a 16-feature NPZ; use val
    pred = predict_split(model, Normalizer(stats), ds, "val")
    assert pred["y_true"].shape == (10, 3, 3)
    assert pred["y_pred"].shape == (10, 3, 3)
    assert np.isfinite(pred["y_pred"]).all()


def test_evaluate_split_metric_keys(trained_eval):
    ds, stats, ckpt = trained_eval
    cfg_path = _write_cfg(ds.parent / "cfg2.json")
    model = load_model_from_config(cfg_path, ckpt)
    result = evaluate_split(model, Normalizer(stats), ds, "val")
    assert result["split"] == "val"
    assert result["samples"] == 10
    for h in ("6h", "12h", "24h"):
        m = result["metrics"][h]
        for key in ("track_error_km_mean", "track_error_km_median",
                    "track_error_km_std", "wind_mae", "wind_rmse"):
            assert key in m
            assert m[key] >= 0


def test_evaluate_never_uses_test(trained_eval):
    """Validation evaluation must not require nor read the TEST NPZ."""
    ds, stats, ckpt = trained_eval
    cfg_path = _write_cfg(ds.parent / "cfg3.json")
    test_npz = ds / "test.npz"
    test_npz.rename(test_npz.with_suffix(".bak"))
    try:
        model = load_model_from_config(cfg_path, ckpt)
        result = evaluate_split(model, Normalizer(stats), ds, "val")
        assert result["samples"] == 10
    finally:
        test_npz.with_suffix(".bak").rename(test_npz)