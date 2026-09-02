""". Tests for the Phase-2 baselines (persistence, movement vector) and geographic metrics.

Standalone runnable:  python -m pytest p4_forecasting/phase2/tests/test_baselines.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ..baselines.movement_vector import MovementVectorBaseline, movement_vector_forecast
from ..baselines.persistence import PersistenceBaseline, persistence_forecast
from ..evaluation.geo_metrics import EARTH_RADIUS_KM, haversine_km
from ..evaluation.metrics import wind_mae, wind_rmse


def _history(last: list, prev6: list) -> np.ndarray:
    """Build a (5,7) history block; only lat/lon/wind matter for baselines."""
    h = np.zeros((5, 7), dtype=np.float32)
    h[3, 0:3] = prev6
    h[4, 0:3] = last
    return h


def test_persistence_exact():
    hist = _history([10.0, 80.0, 120.0], [9.0, 78.0, 115.0])
    f = persistence_forecast(hist)
    assert f.shape == (3, 3)
    for row in f:
        assert math.isclose(row[0], 10.0, rel_tol=1e-6)
        assert math.isclose(row[1], 80.0, rel_tol=1e-6)
        assert math.isclose(row[2], 120.0, rel_tol=1e-6)


def test_persistence_uses_only_last_timestep():
    hist = _history([10.0, 80.0, 120.0], [90.0, 5.0, 300.0])
    f = PersistenceBaseline().predict(hist)
    assert (np.abs(f - np.array([[10, 80, 120]], np.float32)) < 1e-6).all()


def test_movement_extrapolation():
    # t-6h: (10, 80) ; t: (12, 86)  => delta=(2, 6)
    # +6h: (14, 92); +12h: (16, 98); +24h: (20, 110); wind persists 120
    hist = _history([12.0, 86.0, 120.0], [10.0, 80.0, 110.0])
    f = movement_vector_forecast(hist)
    assert f.shape == (3, 3)
    np.testing.assert_allclose(f[:, 0], [14.0, 16.0, 20.0], atol=1e-5)
    np.testing.assert_allclose(f[:, 1], [92.0, 98.0, 110.0], atol=1e-5)
    np.testing.assert_allclose(f[:, 2], [120.0, 120.0, 120.0], atol=1e-5)


def test_movement_lon_wrap():
    # Longitude wraps 0..360: t-6h lon=358, t lon=2  => delta_lon via wrap = +4
    # +6h lon = 6, +12h lon = 10, +24h lon = 18; lat delta = 0
    hist = _history([10.0, 2.0, 100.0], [10.0, 358.0, 90.0])
    f = MovementVectorBaseline().predict(hist)
    np.testing.assert_allclose(f[:, 1], [6.0, 10.0, 18.0], atol=1e-5)


def test_haversine_zero():
    d = haversine_km(10.0, 20.0, 10.0, 20.0)
    assert math.isclose(d, 0.0, abs_tol=1e-9)


def test_haversine_symmetric():
    a = haversine_km(12.0, 34.0, -5.0, 120.0)
    b = haversine_km(-5.0, 120.0, 12.0, 34.0)
    assert math.isclose(a, b, rel_tol=1e-12)


def test_haversine_known_quarter_circumference():
    # Pole to equator along a meridian = quarter circumference
    d = haversine_km(90.0, 0.0, 0.0, 0.0)
    expected = 0.25 * 2 * math.pi * EARTH_RADIUS_KM
    assert math.isclose(d, expected, rel_tol=1e-9)


def test_haversine_known_1deg_lat():
    d = haversine_km(0.0, 0.0, 1.0, 0.0)
    expected = math.pi / 180.0 * EARTH_RADIUS_KM
    assert math.isclose(d, expected, rel_tol=1e-9)


def test_haversine_known_london_paris():
    # London <-> Paris approx 344 km
    d = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert 330.0 < d < 360.0, d


def test_haversine_array():
    a = haversine_km(np.array([0.0, 90.0]), np.array([0.0, 0.0]),
                     np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert a.shape == (2,)
    assert a[0] == 0.0


def _synthetic_actual_predicted():
    actual = np.array([
        [[10.0, 80.0, 100.0], [11.0, 82.0, 110.0], [12.0, 85.0, 120.0]],
        [[20.0, 60.0, 130.0], [21.0, 62.0, 140.0], [22.0, 65.0, 150.0]],
    ], dtype=np.float32)
    predicted = np.array([
        [[10.0, 80.0, 90.0], [11.0, 82.0, 100.0], [12.0, 85.0, 110.0]],
        [[20.0, 60.0, 120.0], [21.0, 62.0, 130.0], [22.0, 65.0, 140.0]],
    ], dtype=np.float32)
    return actual, predicted


def test_wind_mae():
    actual, predicted = _synthetic_actual_predicted()
    mae = wind_mae(actual, predicted)
    np.testing.assert_allclose(mae, [10.0, 10.0, 10.0], atol=1e-5)


def test_wind_rmse():
    actual, predicted = _synthetic_actual_predicted()
    rmse = wind_rmse(actual, predicted)
    np.testing.assert_allclose(rmse, [10.0, 10.0, 10.0], atol=1e-5)


def test_track_error_zero_for_identical():
    from ..evaluation.metrics import track_error_km
    actual, predicted = _synthetic_actual_predicted()
    errors = track_error_km(actual, actual)
    for h in range(3):
        assert float(errors[h].max()) == 0.0