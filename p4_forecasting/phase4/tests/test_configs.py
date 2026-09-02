"""Tests for the six Phase-4 experiment configurations and architecture
selection among EXP002-004."""

from __future__ import annotations

import json

import pytest

from phase4.common import HORIZON_HOURS, FEATURE_NAMES, TARGET_NAMES
from phase4.configs import (
    EXP_BASE,
    concretize_best_experiments,
    make_config,
    select_best_architecture,
    write_config,
    write_initial_configs,
)


def test_each_config_contract_fields():
    for eid in ("EXP001", "EXP002", "EXP003", "EXP004"):
        cfg = make_config(eid)
        assert cfg["experiment_id"] == eid
        assert cfg["input_size"] == 16
        assert cfg["output_size"] == 9
        assert cfg["features"] == 16
        assert cfg["feature_order"] == FEATURE_NAMES
        assert cfg["target_order"] == TARGET_NAMES
        assert cfg["horizons_hours"] == HORIZON_HOURS
        assert cfg["seed"] == 42
        assert cfg["model"] in ("improved_lstm", "gru", "multitask_lstm")


def test_exp005_006_start_unfilled_then_concretized():
    cfg5 = make_config("EXP005")
    assert cfg5["loss"] == "huber"
    assert cfg5["model"] is None
    cfg6 = make_config("EXP006")
    assert cfg6["loss"] == "weighted"
    assert cfg6["track_weight"] == pytest.approx(2.0)
    assert cfg6["intensity_weight"] == pytest.approx(1.0)


def test_select_best_architecture_min_track_mean(scratch):
    val = {
        "EXP002": {"6h": {"track_error_km_mean": 100.0},
                   "12h": {"track_error_km_mean": 200.0},
                   "24h": {"track_error_km_mean": 300.0}},
        "EXP003": {"6h": {"track_error_km_mean": 60.0},
                   "12h": {"track_error_km_mean": 120.0},
                   "24h": {"track_error_km_mean": 180.0}},
        "EXP004": {"6h": {"track_error_km_mean": 500.0},
                   "12h": {"track_error_km_mean": 510.0},
                   "24h": {"track_error_km_mean": 520.0}},
    }
    arch = select_best_architecture(val)
    assert arch == "gru"  # EXP003 lowest mean (120)


def test_select_best_architecture_missing_raises():
    with pytest.raises(RuntimeError):
        select_best_architecture({})


def test_write_initial_and_concretize(scratch):
    cfg_dir = scratch / "configs"
    written = write_initial_configs(cfg_dir)
    assert written == ["EXP001", "EXP002", "EXP003", "EXP004"]
    # idempotent: second call rewrites nothing
    assert write_initial_configs(cfg_dir) == []
    others = concretize_best_experiments(cfg_dir, "improved_lstm")
    assert others == ["EXP005", "EXP006"]
    cfg5 = json.loads((cfg_dir / "EXP005.json").read_text(encoding="utf-8"))
    assert cfg5["model"] == "improved_lstm"
    cfg6 = json.loads((cfg_dir / "EXP006.json").read_text(encoding="utf-8"))
    assert cfg6["model"] == "improved_lstm"
    # second call is a no-op (same content)
    assert concretize_best_experiments(cfg_dir, "improved_lstm") == []


def test_write_config_identical_noop(scratch):
    p = scratch / "cfg.json"
    cfg = make_config("EXP001")
    assert write_config(cfg, p) is True
    assert write_config(cfg, p) is False