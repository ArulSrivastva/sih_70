"""Unit tests for Phase-4 loss functions."""

from __future__ import annotations

import torch

import pytest

from phase4.losses import (
    ForecastHuberLoss,
    ForecastMSELoss,
    WeightedMultiTaskLoss,
    build_criterion,
)


def _pred_target(batch=4):
    torch.manual_seed(0)
    pred = torch.randn(batch, 3, 3)
    target = torch.randn(batch, 3, 3)
    return pred, target


def test_mse_matches_manual():
    pred, target = _pred_target()
    loss = ForecastMSELoss()(pred, target)
    manual = torch.mean((pred - target) ** 2)
    assert loss == pytest.approx(manual.item(), abs=1e-5)


def test_mse_zero_when_equal():
    pred = torch.randn(3, 3, 3)
    assert ForecastMSELoss()(pred, pred) == pytest.approx(0.0)


def test_huber_matches_manual():
    pred, target = _pred_target()
    delta = 1.0
    loss = ForecastHuberLoss(delta=delta)(pred, target)
    diff = (pred - target).abs()
    quad = 0.5 * diff.clamp(max=delta) ** 2
    lin = delta * (diff - 0.5 * delta)
    manual = torch.mean(torch.where(diff <= delta, quad, lin))
    assert loss == pytest.approx(manual.item(), abs=1e-5)


def test_weighted_multitask_weights():
    pred, target = _pred_target()
    criterion = WeightedMultiTaskLoss(
        track_weight=2.0, intensity_weight=1.0,
        horizon_weights=[1.0, 1.0, 1.0], base="mse")
    value = criterion(pred, target)
    assert value >= 0
    assert torch.isfinite(torch.tensor(value))
    # exact formula check
    track_mse = torch.mean((pred[..., :2] - target[..., :2]) ** 2)
    inten_mse = torch.mean((pred[..., 2:3] - target[..., 2:3]) ** 2)
    expected = (2.0 * track_mse + 1.0 * inten_mse) / (2.0 + 1.0)
    assert value == pytest.approx(expected.item(), abs=1e-4)


def test_weighted_multitask_horizon_weights():
    pred, target = _pred_target()
    criterion = WeightedMultiTaskLoss(
        track_weight=1.0, intensity_weight=1.0,
        horizon_weights=[100.0, 1.0, 1.0], base="mse")
    value = criterion(pred, target)
    # exact re-derivation with hw normalized to sum 1
    hw = torch.tensor([100.0, 1.0, 1.0]) / 102.0
    err = (pred - target) ** 2
    mean_b = err.mean(dim=0)                      # (3,3)
    track_h = mean_b[:, :2].mean(dim=1)          # (3,)
    inten_h = mean_b[:, 2]                        # (3,)
    track_loss = torch.dot(hw, track_h)
    intensity_loss = torch.dot(hw, inten_h)
    expected = (track_loss + intensity_loss) / 2.0
    assert value == pytest.approx(expected.item(), abs=1e-4)


def test_weighted_multitask_zero_on_perfect():
    pred = torch.randn(2, 3, 3)
    criterion = WeightedMultiTaskLoss(2.0, 1.0, [1, 1, 1], base="mse")
    assert criterion(pred, pred) == pytest.approx(0.0)


def test_weighted_multitask_huber_base():
    criterion = WeightedMultiTaskLoss(2.0, 1.0, [1, 1, 1], base="huber")
    pred, target = _pred_target()
    value = criterion(pred, target)
    assert torch.isfinite(torch.tensor(value)) and value >= 0


def test_build_criterion_dispatch():
    assert isinstance(build_criterion({"loss": "mse"}), ForecastMSELoss)
    assert isinstance(build_criterion({"loss": "huber", "huber_delta": 0.5}),
                      ForecastHuberLoss)
    c = build_criterion({"loss": "weighted", "track_weight": 2.0,
                         "intensity_weight": 1.0, "horizon_weights": [1, 1, 1],
                         "loss_base": "mse", "huber_delta": 1.0})
    assert isinstance(c, WeightedMultiTaskLoss)
    assert c.track_weight == pytest.approx(2.0)
    with pytest.raises(ValueError):
        build_criterion({"loss": "unknown"})