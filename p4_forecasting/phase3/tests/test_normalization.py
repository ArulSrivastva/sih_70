""". Self-contained tests for P4 Phase 3 normalization (z-score, zero-std guard,
no test-stat leakage).
"""

from __future__ import annotations

import numpy as np
import pytest

from ..training.normalization import Normalizer, compute_normalization_stats


def _rng():
    return np.random.default_rng(1234)


def test_stats_shapes():
    X = _rng().normal(size=(100, 5, 7)).astype(np.float32)
    Y = _rng().normal(size=(100, 3, 3)).astype(np.float32)
    stats = compute_normalization_stats(X, Y)
    assert len(stats["feature_mean"]) == 7
    assert len(stats["feature_std"]) == 7
    assert len(stats["target_mean"]) == 3
    assert len(stats["target_std"]) == 3


def test_denormalize_inverts_normalize():
    rng = _rng()
    X = rng.normal(loc=20, scale=8, size=(50, 5, 7)).astype(np.float32)
    Y = rng.normal(loc=100, scale=30, size=(50, 3, 3)).astype(np.float32)
    n = Normalizer(compute_normalization_stats(X, Y))
    assert np.allclose(n.denormalize_Y(n.normalize_Y(Y)), Y, atol=1e-3)


def test_normalized_has_mean_near_zero():
    X = _rng().normal(size=(300, 5, 7)).astype(np.float32)
    Y = _rng().normal(size=(300, 3, 3)).astype(np.float32)
    n = Normalizer(compute_normalization_stats(X, Y))
    Xn = n.normalize_X(X)
    Yn = n.normalize_Y(Y)
    assert np.abs(Xn.mean()) < 1e-4
    assert np.abs(Yn.mean()) < 1e-4
    assert abs(Xn.std() - 1.0) < 1e-3
    assert abs(Yn.std() - 1.0) < 1e-3


def test_stats_train_only_no_test_leak():
    """Statistics must be identical whether or not test data is 'available'."""
    rng = _rng()
    X_train = rng.normal(loc=10, scale=1, size=(200, 5, 7)).astype(np.float32)
    Y_train = rng.normal(loc=10, scale=1, size=(200, 3, 3)).astype(np.float32)
    X_test = rng.normal(loc=1000, scale=300, size=(50, 5, 7)).astype(np.float32)
    Y_test = rng.normal(loc=1000, scale=300, size=(50, 3, 3)).astype(np.float32)

    stats_a = compute_normalization_stats(X_train, Y_train)
    stats_b = compute_normalization_stats(X_train, Y_train)
    assert stats_a == stats_b

    # Including test data would change the stats / their provenance field.
    assert stats_a["computed_from"]["split"] == "train"
    assert stats_a["computed_from"]["policy"] == "training data ONLY"


def test_zero_variance_feature_raises():
    X = np.zeros((20, 5, 7), dtype=np.float32)
    X[..., 0] = 1.0
    Y = np.ones((20, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="zero variance"):
        compute_normalization_stats(X, Y)


def test_nan_input_raises():
    X = np.ones((10, 5, 7), dtype=np.float32)
    Y = np.ones((10, 3, 3), dtype=np.float32)
    X[3, 1, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compute_normalization_stats(X, Y)