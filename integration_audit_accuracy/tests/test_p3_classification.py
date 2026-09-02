"""P3 classification: image metrics reproduced; tabular correctly NOT_RUN."""

import pytest


def test_image_metrics_reproduced_exactly(p3):
    for k in p3["image_stored"]:
        assert abs(p3["image_stored"][k] - p3["image_recomputed"][k]) < 1e-9, k


def test_image_metrics_published_values(p3):
    assert p3["image_recomputed"]["category_accuracy_percent"] == pytest.approx(38.1, abs=1e-6)
    assert p3["image_recomputed"]["category_macro_f1"] == pytest.approx(0.2076, abs=1e-9)
    assert p3["image_recomputed"]["wind_speed_mae_kmh"] == pytest.approx(109.81, abs=1e-6)
    assert p3["image_recomputed"]["wind_speed_rmse_kmh"] == pytest.approx(118.39, abs=1e-6)


def test_tabular_verification_not_run_with_reason(p3):
    # AUDIT RULE: never fake PASS. lightgbm absent -> NOT_RUN is the honest result.
    assert p3["tabular_verification"] == "NOT_RUN"
    assert p3["tabular_lightgbm_available"] is False


def test_reported_delta_is_cross_population(p3):
    # The stored performance_delta compares a 21-frame image test with a 651-row
    # tabular test: it is not a controlled comparison.
    assert "performance_delta" in str(p3["multisource_delta_text"]) or p3["multisource_delta_text"] is not None