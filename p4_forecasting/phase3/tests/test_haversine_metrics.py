""". Haversine metric regression tests for P4 Phase 3 (reuses Phase-2 implementation).
"""

from __future__ import annotations

import math

import numpy as np

from phase2.evaluation.geo_metrics import EARTH_RADIUS_KM, haversine_km
from phase2.evaluation.metrics import summarize_split, track_error_km


def test_haversine_known_1deg_lat():
    d = haversine_km(0.0, 0.0, 1.0, 0.0)
    assert math.isclose(d, math.pi / 180.0 * EARTH_RADIUS_KM, rel_tol=1e-9)


def test_haversine_known_london_paris():
    d = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert 330.0 < d < 360.0, d


def test_track_error_zero_when_identical():
    actual = np.random.default_rng(1).normal(size=(5, 3, 3)).astype(np.float32)
    errors = track_error_km(actual, actual)
    assert max(float(e.max()) for e in errors) == 0.0


def test_summarize_split_structure():
    rng = np.random.default_rng(2)
    actual = rng.normal(loc=0, scale=10, size=(20, 3, 3)).astype(np.float32)
    predicted = rng.normal(loc=0, scale=10, size=(20, 3, 3)).astype(np.float32)
    out = summarize_split(actual, predicted)
    for hz in ("6h", "12h", "24h"):
        for k in ("track_error_km_mean", "track_error_km_median",
                  "track_error_km_std", "wind_mae", "wind_rmse"):
            assert np.isfinite(out[hz][k])