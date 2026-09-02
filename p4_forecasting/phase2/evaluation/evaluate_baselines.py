"""Evaluate persistence and movement-vector baselines on the CLEAN chronological dataset.

Primary comparison uses the TEST set (per the P4 Phase-2 spec); validation results are
reported separately. All numbers come from actual forward passes — nothing is fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..baselines.movement_vector import MovementVectorBaseline
from ..baselines.persistence import PersistenceBaseline
from .metrics import summarize_split


def _run_model(X: np.ndarray, model) -> np.ndarray:
    """Apply a baseline to a full array of histories -> (N,3,3) forecasts."""
    N = X.shape[0]
    preds = np.empty((N, 3, 3), dtype=np.float32)
    for i in range(N):
        preds[i] = model.predict(X[i])
    return preds


def evaluate_baselines(
    dataset_dir: Path,
    split_counts: Dict[str, Dict[str, int]],
) -> Dict[str, object]:
    """Evaluate both baselines on val and test; returns the full result structure."""
    baseline_names = ["persistence", "movement_vector"]
    baselines = {
        "persistence": PersistenceBaseline(),
        "movement_vector": MovementVectorBaseline(),
    }

    result: Dict[str, object] = {
        "dataset": {
            "name": "canonical_chronological_clean",
            "policy": "CLEAN-only chronological (no non-causal/interpolated cells)",
            "train_cyclones": split_counts["train"]["cyclones"],
            "train_sequences": split_counts["train"]["sequences"],
            "validation_cyclones": split_counts["val"]["cyclones"],
            "validation_sequences": split_counts["val"]["sequences"],
            "test_cyclones": split_counts["test"]["cyclones"],
            "test_sequences": split_counts["test"]["sequences"],
            "feature_order": ["lat", "lon", "wind_speed", "pressure", "sst", "wind_u", "wind_v"],
            "target_order": ["lat", "lon", "wind_speed"],
            "horizons_hours": [6, 12, 24],
            "sst_unit": "degrees Celsius (as delivered; not converted)",
        },
        "primary_split": "test",
        "persistence": {},
        "movement_vector": {},
        "validation_results": {
            "persistence": {},
            "movement_vector": {},
        },
    }

    loaded = {s: np.load(dataset_dir / f"{s}.npz", allow_pickle=True) for s in ("val", "test")}
    X_val, Y_val = loaded["val"]["X"], loaded["val"]["Y"]
    X_test, Y_test = loaded["test"]["X"], loaded["test"]["Y"]

    for name in baseline_names:
        model = baselines[name]
        pred_test = _run_model(X_test, model)
        pred_val = _run_model(X_val, model)

        result[name] = summarize_split(Y_test, pred_test)
        result["validation_results"][name] = summarize_split(Y_val, pred_val)

    return result


def write_baseline_results(data: Dict[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")