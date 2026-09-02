"""Tests for the experiment registry CSV (exactly 22 required columns)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from phase4.registry import (
    REGISTRY_COLUMNS,
    REGISTRY_COLUMN_COUNT,
    build_registry,
    row_from_experiment,
    save_registry,
)


def _sample_summary(**over):
    import numpy as np

    summary = {
        "experiment_id": "EXP001",
        "status": "PASS",
        "config": {
            "model": "improved_lstm", "features": 16, "loss": "mse",
            "hidden_size": 64, "layers": 1, "dropout": 0.0,
            "learning_rate": 0.001, "batch_size": 64,
        },
        "best_epoch": 12,
        "best_val_loss": 0.01,
        "validation_primary_score": 77.5,
        "validation_metrics": {
            "6h": {"track_error_km_mean": 70.0, "wind_mae": 6.0, "wind_rmse": 9.0},
            "12h": {"track_error_km_mean": 80.0, "wind_mae": 8.0, "wind_rmse": 12.0},
            "24h": {"track_error_km_mean": 82.5, "wind_mae": 12.0, "wind_rmse": 16.0},
        },
    }
    summary.update(over)
    return summary


def test_registry_has_exactly_22_required_columns():
    assert REGISTRY_COLUMN_COUNT == 22
    assert REGISTRY_COLUMNS[0] == "experiment_id"
    assert REGISTRY_COLUMNS[-1] == "status"


def test_row_mapping():
    row = row_from_experiment(_sample_summary())
    for col in REGISTRY_COLUMNS:
        assert col in row
    assert row["val_track_6h"] == pytest.approx(70.0)
    assert row["val_track_24h"] == pytest.approx(82.5)
    assert row["val_wind_mae_12h"] == pytest.approx(8.0)
    assert row["val_wind_rmse_24h"] == pytest.approx(16.0)
    assert row["val_primary_score"] == pytest.approx(77.5)
    assert row["status"] == "PASS"


def test_failed_experiment_defaults():
    summary = {"experiment_id": "EXP999", "config": {"loss": "mse"}}
    row = row_from_experiment(summary)
    assert row["status"] == "FAILED"
    assert row["model"] == ""
    assert row["val_track_6h"] is None
    for col in REGISTRY_COLUMNS:
        assert col in row


def test_build_registry_dataframe():
    summaries = [_sample_summary(), {"experiment_id": "EXP002", "config": {}}]
    df = build_registry(summaries)
    assert list(df.columns) == REGISTRY_COLUMNS
    assert len(df) == 2
    assert {"EXP001", "EXP002"} == set(df["experiment_id"])


def test_save_registry_csv_roundtrip(scratch):
    path = scratch / "experiment_registry.csv"
    save_registry(path, [_sample_summary()])
    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.columns) == REGISTRY_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "experiment_id"] == "EXP001"


def test_json_dump_consistency(scratch):
    """The registry survives flush-to-disk identically (no NaN columns dropped)."""
    path = scratch / "experiment_registry.json"
    path.write_text(json.dumps(_sample_summary(), indent=2), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    row = row_from_experiment(loaded)
    assert row["val_track_6h"] == pytest.approx(70.0)
    assert row["status"] == "PASS"