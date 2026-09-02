"""Guard / orchestrator infrastructure tests (WriteGuard, audit helpers,
improvement-percentage math, final-comparison assembly)."""

from __future__ import annotations

import pytest

from phase4.common import PKG_DIR, WriteGuard
from phase4.final_comparison import _pct, build_final_comparison
from phase4.input_audit import (
    AuditContext,
    check_chronology,
    check_clean_mapping,
)


def test_write_guard_rejects_outside_paths():
    guard = WriteGuard(PKG_DIR / "phase4")
    assert guard.check(PKG_DIR / "phase4" / "results") is not None
    with pytest.raises(PermissionError):
        guard.check(PKG_DIR / "canonical")
    with pytest.raises(PermissionError):
        guard.check(PKG_DIR / "phase2")
    with pytest.raises(PermissionError):
        guard.check(PKG_DIR / ".." / "..")


def test_write_guard_join():
    guard = WriteGuard(PKG_DIR / "phase4")
    p = guard.join("results", "sub.txt")
    assert str(p).startswith(str(PKG_DIR / "phase4" / "results"))


def test_improvement_percent_math():
    assert _pct(100.0, 90.0) == pytest.approx(10.0)      # challenger better
    assert _pct(100.0, 110.0) == pytest.approx(-10.0)    # honest negative
    assert _pct(0.0, 5.0) == 0.0


def test_check_chronology_real_manifest():
    manifest = PKG_DIR / "canonical_chrono" / "split_manifest.csv"
    ok, detail = check_chronology(manifest)
    assert ok, detail


def test_check_clean_mapping_real():
    ctx = AuditContext(
        chrono_dir=PKG_DIR / "canonical_chrono",
        canonical_dir=PKG_DIR / "canonical",
        phase2_clean_dir=PKG_DIR / "phase2" / "results" / "canonical_chronological_clean",
        phase2_baseline_json=PKG_DIR / "phase2" / "results" / "baseline_results.json",
        phase3_comparison_json=PKG_DIR / "phase3" / "results" / "model_comparison.json",
        phase3_stats_json=PKG_DIR / "phase3" / "results" / "normalization_stats.json",
        quality_csv=PKG_DIR / "canonical" / "sample_quality.csv",
        p4_dir=PKG_DIR,
    )
    ok, detail = check_clean_mapping(ctx)
    assert ok, detail


def _fake_metrics(tm, wm, wr):
    return {"6h": {"track_error_km_mean": tm, "wind_mae": wm, "wind_rmse": wr},
            "12h": {"track_error_km_mean": tm, "wind_mae": wm, "wind_rmse": wr},
            "24h": {"track_error_km_mean": tm, "wind_mae": wm, "wind_rmse": wr}}


def test_final_comparison_assembly():
    persistence = _fake_metrics(64.0, 3.1, 5.8)
    champion_test = _fake_metrics(40.0, 2.0, 4.0)
    comparison = build_final_comparison(
        phase2_baseline={"persistence": persistence, "movement_vector": persistence},
        phase3_comparison={"test": {"lstm": persistence}},
        champion_test=champion_test,
        champion_val=_fake_metrics(42.0, 2.2, 4.2),
        exp_val_results={"EXP002": {"metrics": _fake_metrics(44.0, 2.4, 4.4)}},
        champion_id="EXP002",
        dataset_info={"train": {"sequences": 1212}},
    )
    assert comparison["champion"] == "EXP002"
    assert comparison["improvement_vs_persistence"]["6h"]["track_error_km_mean_pct"] \
        == pytest.approx(37.5)  # (64-40)/64
    assert comparison["improvement_vs_persistence"]["12h"]["wind_mae_pct"] \
        == pytest.approx(35.4838709677, rel=1e-3)