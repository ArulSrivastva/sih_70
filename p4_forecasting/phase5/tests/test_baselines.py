"""Baseline tests: correctness + parity with Phase-2 + input non-mutation."""

import copy

import numpy as np

from phase5.baselines.persistence import persistence_forecast, PersistenceBaseline
from phase5.baselines.movement_vector import movement_vector_forecast
from phase2.baselines.persistence import persistence_forecast as p2_persist
from phase2.baselines.movement_vector import movement_vector_forecast as p2_mvec


def _hist():
    return np.array([
        [23.30, 68.50, 55.6, 994.0, 28.67, 8.5095, 1.2749],
        [23.40, 68.00, 64.8, 993.0, 28.26, 4.4456, 7.0864],
        [23.40, 67.10, 64.8, 993.0, 28.43, 7.9810, -3.3996],
        [23.60, 66.40, 64.8, 992.0, 28.69, 2.0987, -4.1294],
        [23.50, 65.70, 74.1, 991.0, 28.85, 2.4551, 4.3582],
    ], dtype=np.float32)


def test_persistence_correct():
    h = _hist()
    out = persistence_forecast(h)
    expected = np.tile(h[-1, :3].astype(np.float32), (3, 1))
    assert np.array_equal(out, expected)
    assert out.shape == (3, 3)


def test_persistence_matches_phase2():
    h = _hist()
    assert np.array_equal(persistence_forecast(h), p2_persist(h))


def test_movement_vector_matches_phase2():
    h = _hist()
    assert np.array_equal(movement_vector_forecast(h), p2_mvec(h))


def test_movement_vector_math():
    h = _hist()
    out = movement_vector_forecast(h)
    lat_t, lon_t = h[-1, 0], h[-1, 1]
    dlat = lat_t - h[-2, 0]
    dlon = (lon_t - h[-2, 1] + 180.0) % 360.0 - 180.0
    for i, m in enumerate((1.0, 2.0, 4.0)):
        assert abs(out[i, 0] - (lat_t + m * dlat)) < 1e-4
        assert abs(((out[i, 1] - (lon_t + m * dlon)) % 360.0) % 360.0) < 1e-4
        assert out[i, 2] == h[-1, 2]


def test_baselines_do_not_modify_input():
    h = _hist().copy()
    before = h.copy()
    persistence_forecast(h)
    movement_vector_forecast(h)
    PersistenceBaseline()(h)
    assert np.array_equal(h, before)


def test_baseline_rejects_bad_shape():
    try:
        movement_vector_forecast(np.zeros((4, 7), dtype=np.float32))
        assert False
    except ValueError:
        pass