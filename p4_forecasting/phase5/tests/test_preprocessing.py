"""Preprocessing tests: 7->16 expansion, order, first-step zero policy."""

import numpy as np

from phase5.config import (DERIVED_FEATURES, FEATURE_NAMES, FEATURE_NAMES_RAW,
                           N_FEATURES, N_TARGETS)
from phase5.inference.preprocessing import (check_first_step_zero_fill,
                                            denormalize_targets,
                                            engineer_history,
                                            load_feature_order,
                                            load_normalizer,
                                            normalize_features)
from phase5.inference.input_validation import parse_history
from phase5.service.forecasting_service import ForecastingService


def _hist():
    return np.array([
        [23.30, 68.50, 55.6, 994.0, 28.67, 8.5095, 1.2749],
        [23.40, 68.00, 64.8, 993.0, 28.26, 4.4456, 7.0864],
        [23.40, 67.10, 64.8, 993.0, 28.43, 7.9810, -3.3996],
        [23.60, 66.40, 64.8, 992.0, 28.69, 2.0987, -4.1294],
        [23.50, 65.70, 74.1, 991.0, 28.85, 2.4551, 4.3582],
    ], dtype=np.float32)


def test_7_to_16_expansion():
    feats = engineer_history(_hist())
    assert feats.shape == (5, N_FEATURES)
    assert feats.dtype == np.float32


def test_feature_order_matches_contract(stats_path="phase4/results/normalization_stats.json"):
    order = load_feature_order("phase4/results/normalization_stats.json")
    assert order == FEATURE_NAMES
    assert order[:7] == FEATURE_NAMES_RAW
    assert order[7:] == DERIVED_FEATURES


def test_raw_seven_byte_identical():
    h = _hist()
    feats = engineer_history(h)
    assert np.array_equal(feats[:, :7], h)


def test_first_step_zero_fill():
    feats = engineer_history(_hist())
    assert np.all(feats[0, 7:14] == 0.0)
    assert check_first_step_zero_fill(feats)


def test_environmental_features_nonzero_at_first_step():
    feats = engineer_history(_hist())
    # env wind speed = hypot(u,v) > 0; env direction defined at every step
    assert np.hypot(feats[0, 5], feats[0, 6]) > 0.0
    assert feats[0, 14] > 0.0
    assert 0.0 <= feats[0, 15] < 360.0


def test_normalize_denormalize_roundtrip():
    stats = load_normalizer("phase4/results/normalization_stats.json")
    feats = engineer_history(_hist())
    xn = normalize_features(feats, stats)
    assert xn.shape == (5, 16)
    assert not np.isnan(xn).any()
    # denorm of normalized should round-trip z-scores
    yn = np.array([[0.0, 0.0, 0.0]], dtype=np.float32).repeat(3, axis=0)
    phys = denormalize_targets(yn, stats)
    assert phys.shape == (3, N_TARGETS)
    assert np.allclose(phys, np.array(stats.target_mean, np.float32),
                       atol=0.02)


def test_denormalize_target_shape_enforced():
    stats = load_normalizer("phase4/results/normalization_stats.json")
    try:
        denormalize_targets(np.zeros((2, 3), dtype=np.float32), stats)
        assert False, "should raise"
    except ValueError:
        pass


def test_stats_are_train_only():
    from phase5.inference.preprocessing import inspect_stats
    info = inspect_stats("phase4/results/normalization_stats.json")
    assert info["computed_from"]["split"] == "train"
    assert info["n_train_samples"] == 1212
    assert info["zero_std_features"] == []