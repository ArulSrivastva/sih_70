""". Integration-contract tests for P4 Phase 3 inference.

These require a trained artifact set (checkpoint + stats + config). They are
auto-skipped when the model has not been trained yet (e.g. a bare unit-test run);
run_phase3.py always trains before invoking pytest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ..inference.forecaster import CycloneForecaster, HORIZONS_HOURS

P3_DIR = Path(__file__).resolve().parents[1]
CKPT = P3_DIR / "checkpoints" / "best_lstm.pt"
CONFIG = P3_DIR / "checkpoints" / "model_config.json"
STATS = P3_DIR / "results" / "normalization_stats.json"

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and CONFIG.exists() and STATS.exists()),
    reason="Phase-3 artifacts not trained yet")


@pytest.fixture(scope="module")
def forecaster() -> CycloneForecaster:
    return CycloneForecaster(CKPT, CONFIG, STATS, device="cpu")


def _history() -> np.ndarray:
    rng = np.random.default_rng(7)
    h = rng.normal(loc=[15.0, 88.0, 60.0, 990.0, 28.0, -3.0, 2.0],
                   scale=[2.0, 4.0, 20.0, 15.0, 1.0, 6.0, 6.0], size=(5, 7))
    return h.astype(np.float32)


def test_inference_output_shape_and_horizons(forecaster):
    out = forecaster.forecast(_history())
    fc = out["forecast"]
    assert len(fc) == 3
    assert [f["hours"] for f in fc] == HORIZONS_HOURS == [6, 12, 24]


def test_inference_output_finite(forecaster):
    out = forecaster.forecast(_history())
    for f in out["forecast"]:
        for k in ("latitude", "longitude", "wind_speed_kmh"):
            assert np.isfinite(f[k]), (k, f[k])


def test_inference_contract_keys_and_units(forecaster):
    out = forecaster.forecast(_history())
    assert set(out.keys()) == {"forecast"}
    for f in out["forecast"]:
        assert set(f.keys()) == {"hours", "latitude", "longitude", "wind_speed_kmh"}
        assert isinstance(f["hours"], int)
        assert isinstance(f["latitude"], float)
        assert isinstance(f["longitude"], float)
        assert isinstance(f["wind_speed_kmh"], float)
        assert 0.0 <= f["longitude"] < 360.0
        assert -90.0 <= f["latitude"] <= 90.0


def test_inference_no_future_data_used(forecaster):
    """Structural check: forecast output does not include any future/target fields."""
    out = forecaster.forecast(_history())
    for f in out["forecast"]:
        assert "target" not in f
        assert "y" not in f
        assert "future" not in f


def test_inference_rejects_wrong_shape(forecaster):
    with pytest.raises(ValueError):
        forecaster.forecast(np.zeros((4, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        forecaster.forecast(np.zeros((5, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        forecaster.forecast(np.zeros((1, 5, 7), dtype=np.float32))


def test_inference_rejects_nan_history(forecaster):
    h = _history()
    h[0, 0] = np.nan
    with pytest.raises(ValueError):
        forecaster.forecast(h)


def test_inference_deterministic(forecaster):
    a = forecaster.forecast(_history())
    b = forecaster.forecast(_history())
    for fa, fb in zip(a["forecast"], b["forecast"]):
        assert fa["latitude"] == fb["latitude"]
        assert fa["longitude"] == fb["longitude"]
        assert fa["wind_speed_kmh"] == fb["wind_speed_kmh"]


def test_inference_denormalized_physical_scale(forecaster):
    """Denormalized forecasts must be back in physical units (lat~degrees, wind~km/h)."""
    out = forecaster.forecast(_history())
    for f in out["forecast"]:
        assert -90.0 <= f["latitude"] <= 90.0
        assert 0.0 <= f["wind_speed_kmh"] <= 500.0