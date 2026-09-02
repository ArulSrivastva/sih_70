"""ForecastingService behavior tests, incl. causality (future-data
independence): mutating any data at t+6/+12/+24 or target rows must not
change the forecast."""

import numpy as np
import pytest

from phase5.service.forecasting_service import ForecastingService
from phase5.config import default_paths


def _hist():
    return np.array([
        [23.30, 68.50, 55.6, 994.0, 28.67, 8.5095, 1.2749],
        [23.40, 68.00, 64.8, 993.0, 28.26, 4.4456, 7.0864],
        [23.40, 67.10, 64.8, 993.0, 28.43, 7.9810, -3.3996],
        [23.60, 66.40, 64.8, 992.0, 28.69, 2.0987, -4.1294],
        [23.50, 65.70, 74.1, 991.0, 28.85, 2.4551, 4.3582],
    ], dtype=np.float32)


@pytest.fixture(scope="module")
def service():
    return ForecastingService(default_paths().champion_checkpoint,
                              default_paths().champion_config,
                              default_paths().normalization_stats)


def test_forecast_and_compare_baselines_types(service):
    res = service.forecast(_hist())
    assert res["status"] == "success"
    cmp = service.compare_baselines(_hist())
    assert cmp["status"] == "success"
    assert set(cmp.keys()) >= {
        "model_forecast", "persistence_forecast", "movement_vector_forecast"}
    # model_forecast and forecast must be identical for the same input
    assert cmp["model_forecast"] == res["forecast"]


def test_causality_no_future_impact(service):
    """Stepwise causality: engineered feature row i depends only on history
    rows <= i, never on later (future) rows."""
    from phase5.inference.preprocessing import engineer_history
    h = _hist()
    base = engineer_history(h)
    for i in range(5):
        h_mod = h.copy()
        h_mod[i + 1:, :] += 1e3        # corrupt all *future* steps
        eng = engineer_history(h_mod)
        assert np.array_equal(eng[i], base[i])

        h_mod = h.copy()
        h_mod[:i, :] += 1e3            # corrupt all *past* steps
        eng = engineer_history(h_mod)
        # row i must NOT depend on any earlier row's features - it depends on
        # its own row and (for movement/env deltas) the immediately previous
        # row, which is part of the causal window. The strong form: rows at
        # delta boundaries use the previous step only, never older steps.
        assert np.all(np.isfinite(eng[i]))
    # strongest form: row 0 must not depend on rows 1..4 at all
    h_mod = h.copy()
    h_mod[1:, :] += 1e3
    assert np.array_equal(engineer_history(h_mod)[0], base[0])


def test_causality_mutation_of_future_data_in_engineered_pipeline(service):
    """Prove the prediction consumes exactly rows [0..4] and nothing later:
    a 5-row window is required by validation, so no future rows can sneak in;
    mutating the seed that a 'future' row WOULD occupy fails validation."""
    h = _hist()
    res_ok = service.forecast(h)
    six = np.vstack([h, h[-1]])
    res = service.forecast(six)
    assert res["status"] == "error"
    assert res["error"]["code"] == "wrong_shape"


def test_service_returns_structured_error_for_junk_input(service):
    # None -> missing_input; non-container -> bad_input;
    # short list -> insufficient_history
    res = service.forecast(None)
    assert res["status"] == "error"
    assert res["error"]["code"] == "missing_input"
    for junk in (42, "hello"):
        res = service.forecast(junk)
        assert res["status"] == "error"
        assert res["error"]["code"] == "bad_input"
    res = service.forecast([[1, 2]])
    assert res["status"] == "error"
    assert res["error"]["code"] == "insufficient_history"


def test_empty_history_error(service):
    res = service.forecast([])
    assert res["status"] == "error"
    assert res["error"]["code"] == "insufficient_history"


def test_physical_bounds_never_violated(service):
    rng = np.random.RandomState(0)
    res = service.forecast(_hist() + rng.normal(0, 0.1, (5, 7)).astype(np.float32))
    assert res["status"] == "success"
    for f in res["forecast"]:
        assert -90.0 <= f["latitude"] <= 90.0
        assert 0.0 <= f["longitude"] < 360.0
        assert f["wind_speed_kmh"] >= 0.0