"""Unit tests for Phase-4 feature engineering (locked 16-feature contract)."""

from __future__ import annotations

import numpy as np
import pytest

from phase4.common import (
    DERIVED_FEATURES,
    FEATURE_NAMES,
    FEATURE_NAMES_RAW,
    HISTORY_STEPS,
    N_FEATURES,
)
from phase4.features.feature_engineering import (
    FIRST_STEP_ZERO_FILL_DOC,
    PREDECESSOR_DEPENDENT,
    bearing_degrees,
    engineer_features,
    environmental_wind_direction,
    features_from_history,
    validate_raw,
    wrap_lon_delta,
)
from phase4.features._geo import haversine_km


def test_engineer_features_shape_and_identity(synthetic_raw_history):
    X = synthetic_raw_history
    Xf = engineer_features(X)
    assert Xf.shape == (X.shape[0], HISTORY_STEPS, N_FEATURES)
    assert Xf.shape[-1] == 16
    # the first seven raw columns must be byte-identical to the input
    assert np.array_equal(Xf[:, :, :7], X)


def test_zero_fill_first_step(synthetic_raw_history):
    X = synthetic_raw_history
    Xf = engineer_features(X)
    # i=0 predecessor-dependent columns exactly zero
    for name in PREDECESSOR_DEPENDENT:
        idx = FEATURE_NAMES.index(name)
        assert np.all(Xf[:, 0, idx] == 0.0), f"{name} not zero-filled at i=0"


def test_environmental_features_not_zero_filled(synthetic_raw_history):
    X = synthetic_raw_history
    Xf = engineer_features(X)
    i_speed = FEATURE_NAMES.index("environmental_wind_speed")
    i_dir = FEATURE_NAMES.index("environmental_wind_direction")
    u = X[:, 0, 5]
    v = X[:, 0, 6]
    # computed from current timestep, NOT zero-filled
    assert np.allclose(Xf[:, 0, i_speed], np.hypot(u, v), rtol=1e-5, atol=1e-5)
    assert np.allclose(Xf[:, 0, i_dir], environmental_wind_direction(u, v),
                       rtol=1e-5, atol=1e-5)
    # nonzero even though it is the first step
    assert np.any(Xf[:, 0, i_speed] > 0)


def test_delta_lon_wrap_across_360(synthetic_raw_history):
    X = synthetic_raw_history
    Xf = engineer_features(X)
    # sample 1 (row index 1) crosses 359 -> 1 at steps 2->3
    d = Xf[1, 3, FEATURE_NAMES.index("delta_lon")]
    assert d == pytest.approx(2.0, abs=1e-4)  # 1 - 359 wrapped -> +2, not -358


def test_wrap_lon_delta_function():
    assert wrap_lon_delta(np.array([359.0]), np.array([1.0]))[0] == pytest.approx(2.0)
    assert wrap_lon_delta(np.array([1.0]), np.array([359.0]))[0] == pytest.approx(-2.0)
    assert wrap_lon_delta(np.array([100.0]), np.array([105.0]))[0] == pytest.approx(5.0)


def test_movement_speed_matches_haversine():
    rng = np.random.RandomState(3)
    lat = rng.uniform(8, 22, size=(4,))
    lon = rng.uniform(50, 90, size=(4,))
    lat2 = lat + 0.3
    lon2 = lon + 0.3
    km = haversine_km(lat, lon, lat2, lon2)
    expected = km / 6.0  # km/h over a 6-hour step
    X = np.zeros((4, 5, 7), dtype=np.float32)
    X[:, 3, 0] = lat  # source coordinates at step 3
    X[:, 3, 1] = lon
    X[:, 4, 0] = lat2  # destination at step 4
    X[:, 4, 1] = lon2
    Xf = engineer_features(X)
    got = Xf[:, 4, FEATURE_NAMES.index("movement_speed")]
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-3)


def test_bearing_range_and_north():
    # moving due east from equator -> bearing 90
    b = bearing_degrees(np.zeros(1), np.zeros(1), np.zeros(1), np.array([1.0]))
    assert b[0] == pytest.approx(90.0, abs=1e-3)
    # moving due north -> bearing 0 (mod 360, tolerant of -1e-7 -> 359.9999)
    b2 = bearing_degrees(np.array([10.0]), np.array([0.0]),
                         np.array([11.0]), np.array([0.0]))
    assert b2[0] % 360.0 == pytest.approx(0.0, abs=1e-2)


def test_wind_direction_convention():
    # u=1 (easterly), v=0 -> wind blows toward east -> FROM WEST = 270
    assert environmental_wind_direction(np.array([1.0]), np.array([0.0]))[0] == \
        pytest.approx(270.0)
    # u=0, v=1 (northerly) -> wind blows toward north -> FROM SOUTH = 180
    assert environmental_wind_direction(np.array([0.0]), np.array([1.0]))[0] == \
        pytest.approx(180.0, abs=1e-2)
    # u=1, v=1 (northeasterly) -> FROM SW = 225
    assert environmental_wind_direction(np.array([1.0]), np.array([1.0]))[0] == \
        pytest.approx(225.0, abs=1e-2)


def test_no_nan_inf(synthetic_raw_history):
    Xf = engineer_features(synthetic_raw_history)
    assert not np.isnan(Xf).any()
    assert not np.isinf(Xf).any()


def test_validate_raw_rejects_bad_inputs():
    rng = np.random.RandomState(1)
    with pytest.raises(ValueError):
        validate_raw(rng.uniform(size=(10, 6, 7)).astype(np.float32))   # wrong steps
    with pytest.raises(ValueError):
        validate_raw(rng.uniform(size=(10, 5, 6)).astype(np.float32))   # wrong features
    bad = rng.uniform(size=(10, 5, 7)).astype(np.float32)
    bad[3, 2, 0] = np.nan
    with pytest.raises(ValueError):
        validate_raw(bad)


def test_features_from_history(synthetic_raw_history):
    h = synthetic_raw_history[0]  # (5,7)
    out = features_from_history(h)
    assert out.shape == (HISTORY_STEPS, N_FEATURES)
    assert np.array_equal(out[:, :7], h)
    with pytest.raises(ValueError):
        features_from_history(h[None, ...])  # (1,5,7) is not (5,7)


def test_derived_feature_names_contract():
    assert len(FEATURE_NAMES) == 16
    assert FEATURE_NAMES[:7] == FEATURE_NAMES_RAW
    assert FEATURE_NAMES[7:] == DERIVED_FEATURES
    assert FIRST_STEP_ZERO_FILL_DOC  # documented policy is present