"""Unit tests for the deterministic Phase-4 trainer."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from phase4.models import ImprovedLSTM
from phase4.training.train import TrainConfig, run_training
from tests.conftest import make_feature_npz, make_normalization_stats


def _small_cfg(**over):
    base = dict(seed=42, learning_rate=1e-3, batch_size=8,
                max_epochs=3, patience=2, device="cpu", force_retrain=True)
    base.update(over)
    return TrainConfig(**base)


def _run(dataset_dir, checkpoint, history, stats, cfg=None, force=True):
    cfg = cfg or _small_cfg(force_retrain=force)
    return run_training(
        dataset_dir=dataset_dir, stats=stats,
        checkpoint_path=checkpoint, history_path=history, cfg=cfg,
        build_model=lambda: ImprovedLSTM(input_size=16, hidden_size=8,
                                         num_layers=1, output_size=9),
        build_criterion=lambda: torch.nn.MSELoss())


def test_training_produces_checkpoint_and_history(scratch):
    ds = make_feature_npz(scratch / "ds")
    stats = make_normalization_stats()
    ckpt = scratch / "ckpt.pt"
    hist = scratch / "training_history.json"
    out = _run(ds, ckpt, hist, stats)
    assert out.reused is False
    assert out.best_epoch >= 1
    assert out.best_val_loss is not None and np.isfinite(out.best_val_loss)
    assert ckpt.exists()
    assert hist.exists()
    payload = json.loads(hist.read_text(encoding="utf-8"))
    assert payload["best_val_loss"] == pytest.approx(out.best_val_loss)
    assert out.parameter_count > 0


def test_training_never_requires_test_split(scratch):
    ds = make_feature_npz(scratch / "ds")
    (ds / "test.npz").unlink()
    ckpt = scratch / "ckpt.pt"
    hist = scratch / "training_history.json"
    out = _run(ds, ckpt, hist, make_normalization_stats())
    assert out.reused is False


def test_training_deterministic_with_force_retrain(scratch):
    ds = make_feature_npz(scratch / "ds", n_train=32, n_val=8)
    stats_clean = make_normalization_stats()
    a = _run(ds, scratch / "a.pt", scratch / "a.json", stats_clean)
    b = _run(ds, scratch / "b.pt", scratch / "b.json", stats_clean,
             _small_cfg(force_retrain=True))
    assert a.best_val_loss == pytest.approx(b.best_val_loss, abs=1e-6)
    ha = json.loads((scratch / "a.json").read_text(encoding="utf-8"))["history"]
    hb = json.loads((scratch / "b.json").read_text(encoding="utf-8"))["history"]
    assert ha == hb


def test_checkpoint_reuse_is_idempotent(scratch):
    ds = make_feature_npz(scratch / "ds", n_train=16, n_val=4)
    stats = make_normalization_stats()
    ckpt = scratch / "ckpt.pt"
    hist = scratch / "history.json"
    first = _run(ds, ckpt, hist, stats)
    mtime_before = ckpt.stat().st_mtime_ns
    second = _run(ds, ckpt, hist, stats, force=False)
    assert second.reused is True
    assert mtime_before == ckpt.stat().st_mtime_ns
    assert second.best_val_loss == pytest.approx(first.best_val_loss)


def test_early_stopping_stops_before_max_epochs(scratch):
    """lr=0 forces a val-loss plateau at epoch 1, so patience must stop early."""
    ds = make_feature_npz(scratch / "ds", n_train=8, n_val=4)
    stats = make_normalization_stats()
    ckpt = scratch / "ckpt.pt"
    hist = scratch / "history.json"
    out = _run(ds, ckpt, hist, stats, _small_cfg(max_epochs=50, patience=2,
                                                 learning_rate=0.0))
    assert out.epochs_run < 50
    assert out.epochs_run == 3  # best at epoch 1 + patience hits at epoch 3