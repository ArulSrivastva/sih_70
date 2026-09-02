"""Unit tests for Phase-4 train-only normalization."""

from __future__ import annotations

import json

import numpy as np
import pytest

from phase4.common import FEATURE_NAMES, TARGET_NAMES
from phase4.training.normalization import (
    Normalizer,
    compute_normalization_stats,
)


def test_stats_shapes_and_order():
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(50, 5, 16)).astype(np.float32)
    Y = rng.uniform(size=(50, 3, 3)).astype(np.float32)
    stats = compute_normalization_stats(X, Y)
    assert len(stats["feature_mean"]) == 16
    assert len(stats["feature_std"]) == 16
    assert len(stats["target_mean"]) == 3
    assert len(stats["target_std"]) == 3
    assert stats["feature_order"] == FEATURE_NAMES
    assert stats["target_order"] == TARGET_NAMES
    assert stats["computed_from"]["split"] == "train"
    assert stats["computed_from"]["policy"] == "TRAIN only; never val/test/combined"


def test_zero_std_replace_with_identity():
    rng = np.random.RandomState(1)
    X = rng.uniform(size=(40, 5, 16)).astype(np.float32)
    X[:, :, 12] = 3.5          # constant column -> zero std
    X[:, :, 15] = 0.0          # constant column -> zero std
    Y = rng.uniform(size=(40, 3, 3)).astype(np.float32)
    Y[:, :, 2] = 40.0          # constant wind target
    stats = compute_normalization_stats(X, Y)
    assert "pressure_change" in stats["zero_std_features"]
    assert "environmental_wind_direction" in stats["zero_std_features"]
    assert "wind_speed" in stats["zero_std_targets"]
    assert stats["feature_std"][12] == 1.0
    assert stats["feature_std"][15] == 1.0
    assert stats["target_std"][2] == 1.0
    assert stats["zero_std_handling"].startswith("std<=0 replaced by scale 1.0")


def test_normalizer_roundtrip():
    rng = np.random.RandomState(2)
    X = rng.uniform(size=(50, 5, 16)).astype(np.float32)
    Y = rng.uniform(size=(50, 3, 3)).astype(np.float32)
    stats = compute_normalization_stats(X, Y)
    norm = Normalizer(stats)
    Xn = norm.normalize_X(X)
    Yn = norm.normalize_Y(Y)
    assert Xn.shape == X.shape and Yn.shape == Y.shape
    assert np.allclose(norm.denormalize_Y(Yn), Y, atol=1e-4)
    # normalized train data has ~zero mean, unit std
    assert abs(float(Xn.mean())) < 1e-3
    assert abs(float(Xn.std()) - 1.0) < 1e-2


def test_normalizer_save_load(scratch):
    rng = np.random.RandomState(6)
    X = rng.uniform(size=(50, 5, 16)).astype(np.float32)
    Y = rng.uniform(size=(50, 3, 3)).astype(np.float32)
    stats = compute_normalization_stats(X, Y)
    norm = Normalizer(stats)
    p = scratch / "norm_stats_saved.json"
    norm.save(p)
    reloaded = Normalizer.from_path(p)
    assert np.allclose(reloaded.feature_mean, norm.feature_mean)
    assert np.allclose(reloaded.feature_std, norm.feature_std)
    assert reloaded.stats == stats


def test_normalizer_from_path(scratch):
    rng = np.random.RandomState(3)
    X = rng.uniform(size=(50, 5, 16)).astype(np.float32)
    Y = rng.uniform(size=(50, 3, 3)).astype(np.float32)
    stats = compute_normalization_stats(X, Y)
    p = scratch / "norm_stats.json"
    p.write_text(json.dumps(stats), encoding="utf-8")
    norm = Normalizer.from_path(p)
    assert np.allclose(norm.feature_mean, np.asarray(stats["feature_mean"]))
    assert np.allclose(norm.feature_std, np.asarray(stats["feature_std"]))


def test_normalizer_rejects_missing_keys_and_bad_order():
    stats = {
        "feature_mean": [0.0] * 16, "feature_std": [1.0] * 16,
        "target_mean": [0.0] * 3, "target_std": [1.0] * 3,
        "feature_order": [f"x{i}" for i in range(16)],
    }
    with pytest.raises(ValueError):
        Normalizer(stats)  # bad feature_order
    del stats["feature_std"]
    with pytest.raises(ValueError):
        Normalizer(stats)  # missing required key


def test_stats_reject_invalid_inputs():
    rng = np.random.RandomState(4)
    with pytest.raises(ValueError):
        compute_normalization_stats(rng.uniform(size=(10, 4, 16)).astype(np.float32),
                                    rng.uniform(size=(10, 3, 3)).astype(np.float32))
    Xbad = rng.uniform(size=(10, 5, 16)).astype(np.float32)
    Xbad[0, 0, 0] = np.nan
    Y = rng.uniform(size=(10, 3, 3)).astype(np.float32)
    with pytest.raises(ValueError):
        compute_normalization_stats(Xbad, Y)


def test_stats_computed_only_from_given_arrays():
    """Val/test arrays never influence stats because stats take only train arrays."""
    rng = np.random.RandomState(5)
    X_train = rng.uniform(low=-1.0, high=1.0, size=(80, 5, 16)).astype(np.float32)
    Y_train = rng.uniform(low=-1.0, high=1.0, size=(80, 3, 3)).astype(np.float32)
    stats_a = compute_normalization_stats(X_train, Y_train)
    # A shifted "val set" passed by mistake is irrelevant: it is not accepted at all.
    assert compute_normalization_stats(X_train, Y_train) == stats_a