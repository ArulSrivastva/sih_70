"""Build the CLEAN-only chronological forecasting dataset for P4 Phase 2.

Reads the Phase-1 canonical artifacts (read-only):
    p4_forecasting/canonical_chrono/{train,val,test}.npz + *_metadata.csv
    p4_forecasting/canonical/sample_quality.csv

and writes a filtered CLEAN-only copy under
    p4_forecasting/phase2/results/canonical_chronological_clean/

No values are imputed, interpolated, normalized, or transformed. Only rows whose
Phase-1 quality_status == "CLEAN" (no non-causal / interpolated / filled cells)
are retained. Source NPZ/CSV files are never overwritten.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

# Feature / target order documented in the Phase-1 DATA CONTRACT.
FEATURES = ["lat", "lon", "wind_speed", "pressure", "sst", "wind_u", "wind_v"]
TARGETS = ["lat", "lon", "wind_speed"]
SPLITS = ["train", "val", "test"]

# Phase-1 expected CLEAN-only counts (verified against canonical_chrono + sample_quality).
EXPECTED_CLEAN = {
    "train": {"sequences": 1212, "cyclones": 57},
    "val": {"sequences": 231, "cyclones": 13},
    "test": {"sequences": 198, "cyclones": 10},
}


def build_clean_chronological_dataset(
    chrono_dir: Path,
    quality_csv_path: Path,
    out_dir: Path,
) -> Dict[str, object]:
    """Build CLEAN-only npz + metadata for each split under out_dir.

    Returns a summary dict with paths and counts.
    """
    time_str = datetime.now(timezone.utc)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quality = pd.read_csv(quality_csv_path)
    if {"cyclone_id", "t_zero", "quality_status"} - set(quality.columns):
        raise ValueError(f"sample_quality.csv missing expected columns: {quality.columns.tolist()}")

    summary = {
        "built_from": {
            "chrono_dir": str(chrono_dir),
            "quality_csv": str(quality_csv_path),
        },
        "created_at": str(time_str.isoformat()),
        "policy": "keep only rows with quality_status == CLEAN; no value changes",
        "splits": {},
    }

    for split in SPLITS:
        npz_path = chrono_dir / f"{split}.npz"
        meta_path = chrono_dir / f"{split}_metadata.csv"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing canonical source: {npz_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing canonical source: {meta_path}")

        loaded = np.load(npz_path, allow_pickle=True)
        X = np.asarray(loaded["X"], dtype=np.float32)
        Y = np.asarray(loaded["Y"], dtype=np.float32)
        features = [str(x) for x in loaded["features"]]
        targets = [str(x) for x in loaded["targets"]]
        meta = pd.read_csv(meta_path)

        if list(features) != FEATURES:
            raise ValueError(f"{split}: feature order mismatch {features}")
        if list(targets) != TARGETS:
            raise ValueError(f"{split}: target order mismatch {targets}")
        if X.shape[0] != Y.shape[0] or X.shape[0] != meta.shape[0]:
            raise ValueError(f"{split}: inconsistent row counts X/Y/meta")

        merged = meta.merge(
            quality[["cyclone_id", "t_zero", "quality_status", "quality_detail"]],
            on=["cyclone_id", "t_zero"],
            how="left",
        )
        n_unmatched = int(merged["quality_status"].isna().sum())
        if n_unmatched:
            raise ValueError(f"{split}: {n_unmatched} sequences have no quality flag!")

        clean_mask = merged["quality_status"] == "CLEAN"
        X_clean = X[clean_mask]
        Y_clean = Y[clean_mask]
        meta_clean = merged[clean_mask].reset_index(drop=True)

        out_npz = out_dir / f"{split}.npz"
        out_meta = out_dir / f"{split}_metadata.csv"
        np.savez_compressed(out_npz, X=X_clean, Y=Y_clean,
                            features=np.array(FEATURES),
                            targets=np.array(TARGETS))
        meta_clean.to_csv(out_meta, index=False)

        summary["splits"][split] = {
            "source_sequences": int(X.shape[0]),
            "clean_sequences": int(X_clean.shape[0]),
            "cyclones": int(meta_clean["cyclone_id"].nunique()),
            "npz": str(out_npz),
            "metadata": str(out_meta),
            "unmatched_quality_flags": n_unmatched,
            "expected": EXPECTED_CLEAN[split],
        }

    return summary


def load_clean_splits(dataset_dir: Path) -> Dict[str, Dict[str, object]]:
    """Load the built CLEAN-only npz + metadata for all splits (validation helper)."""
    out: Dict[str, Dict[str, object]] = {}
    for split in SPLITS:
        npz = np.load(dataset_dir / f"{split}.npz", allow_pickle=True)
        meta = pd.read_csv(dataset_dir / f"{split}_metadata.csv")
        out[split] = {"X": npz["X"], "Y": npz["Y"], "meta": meta,
                      "features": [str(x) for x in npz["features"]],
                      "targets": [str(x) for x in npz["targets"]]}
    return out