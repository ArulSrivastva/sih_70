"""Phase-4 experiment registry (results/experiment_registry.csv).

Exact 22 required columns; extra columns allowed.  Failed experiments MUST stay
in the registry with status=FAILED - never hidden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REGISTRY_COLUMNS = [
    "experiment_id", "model", "features", "loss", "hidden_size", "layers",
    "dropout", "learning_rate", "batch_size", "best_epoch", "best_val_loss",
    "val_primary_score",
    "val_track_6h", "val_track_12h", "val_track_24h",
    "val_wind_mae_6h", "val_wind_mae_12h", "val_wind_mae_24h",
    "val_wind_rmse_6h", "val_wind_rmse_12h", "val_wind_rmse_24h",
    "status",
]

REGISTRY_COLUMN_COUNT = len(REGISTRY_COLUMNS)  # exactly 22


def row_from_experiment(summary: Dict[str, object]) -> Dict[str, object]:
    cfg = summary.get("config", {}) or {}
    val = summary.get("validation_metrics", {}) or {}
    row: Dict[str, object] = {
        "experiment_id": summary.get("experiment_id", ""),
        "model": cfg.get("model", ""),
        "features": cfg.get("features", 16),
        "loss": cfg.get("loss", ""),
        "hidden_size": cfg.get("hidden_size", ""),
        "layers": cfg.get("layers", ""),
        "dropout": cfg.get("dropout", ""),
        "learning_rate": cfg.get("learning_rate", ""),
        "batch_size": cfg.get("batch_size", ""),
        "best_epoch": summary.get("best_epoch", None),
        "best_val_loss": summary.get("best_val_loss", None),
        "val_primary_score": summary.get("validation_primary_score", None),
        "val_track_6h": val.get("6h", {}).get("track_error_km_mean", None),
        "val_track_12h": val.get("12h", {}).get("track_error_km_mean", None),
        "val_track_24h": val.get("24h", {}).get("track_error_km_mean", None),
        "val_wind_mae_6h": val.get("6h", {}).get("wind_mae", None),
        "val_wind_mae_12h": val.get("12h", {}).get("wind_mae", None),
        "val_wind_mae_24h": val.get("24h", {}).get("wind_mae", None),
        "val_wind_rmse_6h": val.get("6h", {}).get("wind_rmse", None),
        "val_wind_rmse_12h": val.get("12h", {}).get("wind_rmse", None),
        "val_wind_rmse_24h": val.get("24h", {}).get("wind_rmse", None),
        "status": summary.get("status", "FAILED"),
    }
    # make sure we never drop a column
    for col in REGISTRY_COLUMNS:
        if col not in row:
            row[col] = None
    return {col: row[col] for col in REGISTRY_COLUMNS}


def build_registry(summaries: List[Dict[str, object]]) -> pd.DataFrame:
    rows = [row_from_experiment(s) for s in summaries]
    df = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[REGISTRY_COLUMNS]


def save_registry(path: Path, summaries: List[Dict[str, object]]) -> pd.DataFrame:
    df = build_registry(summaries)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df