"""Baseline comparison for P4 Phase 3.

Loads the Phase-2 baseline results and the Phase-3 LSTM test/validation metrics,
and produces model_comparison.json with computed (never hand-typed) improvement
percentages:

    improvement % = (baseline_error - lstm_error) / baseline_error * 100
    positive = LSTM better, negative = LSTM worse.

Track-error comparison uses mean track error (km). Wind comparison uses wind MAE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

HORIZONS = ["6h", "12h", "24h"]


def _pct(baseline: float, lstm: float) -> float:
    if baseline == 0:
        return 0.0
    return float((baseline - lstm) / abs(baseline) * 100.0)


def build_model_comparison(
    baseline_path: Path,
    evaluation_results: Dict[str, object],
    dataset_info: Dict[str, object],
) -> Dict[str, object]:
    """Assemble the three-way comparison on the TEST split (primary)."""
    with open(baseline_path, "r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    lstm_test = evaluation_results["test"]
    lstm_val = evaluation_results["val"]

    test_models = {
        "persistence": baseline["persistence"],
        "movement_vector": baseline["movement_vector"],
        "lstm": lstm_test,
    }
    val_models = {
        "persistence": baseline["validation_results"]["persistence"],
        "movement_vector": baseline["validation_results"]["movement_vector"],
        "lstm": lstm_val,
    }

    comparison: Dict[str, object] = {
        "dataset": dataset_info,
        "primary_split": "test",
        "test": test_models,
        "validation": val_models,
        "improvement_vs_persistence": {},
        "improvement_vs_movement_vector": {},
        "notes": (
            "improvement % = (baseline - lstm)/|baseline| * 100; positive means "
            "LSTM better. Track error uses mean km; wind uses MAE."),
    }

    for hz in HORIZONS:
        p = baseline["persistence"][hz]
        mv = baseline["movement_vector"][hz]
        l = lstm_test[hz]
        comparison["improvement_vs_persistence"][hz] = {
            "track_error_km_mean_pct": _pct(p["track_error_km_mean"], l["track_error_km_mean"]),
            "wind_mae_pct": _pct(p["wind_mae"], l["wind_mae"]),
        }
        comparison["improvement_vs_movement_vector"][hz] = {
            "track_error_km_mean_pct": _pct(mv["track_error_km_mean"], l["track_error_km_mean"]),
            "wind_mae_pct": _pct(mv["wind_mae"], l["wind_mae"]),
        }

    return comparison


def save_model_comparison(comparison: Dict[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")