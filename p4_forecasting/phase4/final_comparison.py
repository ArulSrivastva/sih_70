"""Final comparison artifact (results/FINAL_COMPARISON.json).

Champion vs Phase-2 persistence / movement-vector on TEST plus Phase-3 LSTM test
metrics, with computed (never fabricated) improvement percentages:

    improvement % = (baseline - challenger) / |baseline| * 100
    positive = challenger better, negative = challenger worse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .common import HORIZONS


def _pct(baseline: float, challenger: float) -> float:
    if baseline == 0:
        return 0.0
    return float((baseline - challenger) / abs(baseline) * 100.0)


def build_final_comparison(
    phase2_baseline: Dict[str, object],
    phase3_comparison: Dict[str, object],
    champion_test: Dict[str, object],
    champion_val: Dict[str, object],
    exp_val_results: Dict[str, Dict[str, object]],
    champion_id: str,
    dataset_info: Dict[str, object],
) -> Dict[str, object]:
    def horizon_pct(base_metrics: Dict[str, object], chall_metrics: Dict[str, object]) -> Dict[str, object]:
        out = {}
        for hz in HORIZONS:
            out[hz] = {
                "track_error_km_mean_pct": _pct(base_metrics[hz]["track_error_km_mean"],
                                               chall_metrics[hz]["track_error_km_mean"]),
                "wind_mae_pct": _pct(base_metrics[hz]["wind_mae"],
                                     chall_metrics[hz]["wind_mae"]),
                "wind_rmse_pct": _pct(base_metrics[hz]["wind_rmse"],
                                      chall_metrics[hz]["wind_rmse"]),
            }
        return out

    persistence_test = phase2_baseline["persistence"]
    movement_test = phase2_baseline["movement_vector"]
    lstm_test = phase3_comparison["test"]["lstm"]

    comparison: Dict[str, object] = {
        "dataset": dataset_info,
        "primary_split": "test",
        "champion": champion_id,
        "baselines_test": {
            "persistence": persistence_test,
            "movement_vector": movement_test,
            "phase3_lstm": lstm_test,
        },
        "champion_test": champion_test,
        "champion_validation": champion_val,
        "experiments_validation": exp_val_results,
        "improvement_vs_persistence": horizon_pct(persistence_test, champion_test),
        "improvement_vs_movement_vector": horizon_pct(movement_test, champion_test),
        "improvement_vs_phase3_lstm": horizon_pct(lstm_test, champion_test),
        "improvement_note": (
            "improvement % = (baseline - challenger)/|baseline| * 100; positive = "
            "challenger better; track error uses mean km; wind uses MAE/RMSE. "
            "Negative values mean the baseline beats the Phase-4 champion and are "
            "reported honestly, consistent with Phase-3 reporting."),
    }
    return comparison


def save_final_comparison(comparison: Dict[str, object], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")