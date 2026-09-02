"""Build the Phase-4 causal generic 16-feature dataset(s) from the CLEAN source.

Source (read-only)
------------------
* ``phase2/results/canonical_chronological_clean/{train,val,test}.npz`` +
  ``*_metadata.csv`` -- the authoritative Phase-2 CLEAN-only chronological set
  (Phase-2 policy is authoritative; random reshuffling and split relocation are
  forbidden).  Values are never repaired or modified.
* ``canonical_chrono/*.npz`` + ``sample_quality.csv`` -- read transiently to
  reconstruct the original per-sample index for provenance (traceability).

Output (write only under phase4/results/feature_dataset/)
----------------------------------------------------------
* ``{train,val,test}.npz`` with ``X`` (N,5,16), ``Y`` (N,3,3), feature/target
  name arrays and horizons.
* ``{train,val,test}_metadata.csv`` adding ``source_split`` and
  ``original_sample_index`` to the full Phase-2 provenance row.
* ``results/FEATURE_CONTRACT.md`` and ``features/FEATURE_ENGINEERING_REPORT.md``
  documenting the contract, formulas, units and verification.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..common import (
    DERIVED_FEATURES,
    EXPECTED_CLEAN,
    FEATURE_NAMES,
    FEATURE_NAMES_RAW,
    TARGET_NAMES,
)
from ..features.feature_engineering import (
    FIRST_STEP_ZERO_FILL_DOC,
    engineer_features,
    validate_raw,
)

SPLITS = ["train", "val", "test"]


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def build_one_split(
    split: str,
    clean_dir: Path,
    chrono_dir: Path,
    quality_csv: Path,
    out_dir: Path,
) -> Dict[str, object]:
    npz_path = clean_dir / f"{split}.npz"
    meta_path = clean_dir / f"{split}_metadata.csv"
    chrono_npz = chrono_dir / f"{split}.npz"
    chrono_meta = chrono_dir / f"{split}_metadata.csv"
    for p in (npz_path, meta_path, chrono_npz, chrono_meta, quality_csv):
        if not p.exists():
            raise FileNotFoundError(f"missing Phase-4 feature source: {p}")

    loaded = np.load(npz_path, allow_pickle=True)
    X = np.asarray(loaded["X"], dtype=np.float32)
    Y = np.asarray(loaded["Y"], dtype=np.float32)
    X = validate_raw(X)
    if Y.ndim != 3 or Y.shape[1:] != (3, 3):
        raise ValueError(f"{split}: Y must be (N,3,3); got {Y.shape}")
    meta = pd.read_csv(meta_path)

    X_feat = engineer_features(X)

    # Reconstruct original (canonical_chrono) sample index of each CLEAN row.
    chrono_loaded = np.load(chrono_npz, allow_pickle=True)
    Xc = np.asarray(chrono_loaded["X"], dtype=np.float32)
    chrono_meta = pd.read_csv(chrono_meta)
    quality = pd.read_csv(quality_csv)
    merged = chrono_meta.merge(
        quality[["cyclone_id", "t_zero", "quality_status"]],
        on=["cyclone_id", "t_zero"], how="left",
    )
    if merged["quality_status"].isna().any():
        raise ValueError(f"{split}: canonical_chrono rows without quality flag")
    clean_pos = np.where(merged["quality_status"].to_numpy() == "CLEAN")[0]
    if clean_pos.size != X_feat.shape[0]:
        raise ValueError(
            f"{split}: CLEAN count from canonical_chrono ({clean_pos.size}) != "
            f"Phase-2 clean rows ({X_feat.shape[0]})")

    out_meta = meta.reset_index(drop=True).copy()
    out_meta["source_split"] = split
    out_meta["original_sample_index"] = clean_pos.astype(int)

    # provenance sanity: origin in canonical_chrono must match, in order
    if Xc.shape[0] != chrono_meta.shape[0]:
        raise ValueError(f"{split}: canonical_chrono npz/meta row mismatch")
    if not _arrays_equal(X[:, -1, :3], Xc[clean_pos, -1, :3].astype(np.float32)):
        raise ValueError(f"{split}: CLEAN origin rows do not match canonical_chrono order")

    # light target consistency with metadata columns
    tgt = np.stack(
        [out_meta[f"target_{h}h_{t}"].to_numpy(float)
         for h in ("6", "12", "24") for t in ("lat", "lon", "wind")],
        axis=-1,
    ).reshape(-1, 3, 3).astype(np.float32)
    if not _arrays_equal(tgt, Y):
        raise ValueError(f"{split}: metadata target columns disagree with npz Y")

    out_npz = out_dir / f"{split}.npz"
    out_csv = out_dir / f"{split}_metadata.csv"

    wrote = False
    if out_npz.exists() and out_csv.exists():
        old = np.load(out_npz, allow_pickle=True)
        old_meta = pd.read_csv(out_csv)
        same = (_arrays_equal(old["X"], X_feat)
                and _arrays_equal(old["Y"], Y)
                and list(old["features"]) == FEATURE_NAMES
                and list(old["targets"]) == TARGET_NAMES
                and old_meta[out_meta.columns.tolist()].equals(out_meta))
        if same:
            wrote = False
            print(f"[feature-dataset] {split}: existing output reused (content identical)")
        else:
            np.savez_compressed(out_npz, X=X_feat, Y=Y,
                                features=np.array(FEATURE_NAMES),
                                targets=np.array(TARGET_NAMES),
                                horizons=np.array([6, 12, 24]))
            out_meta.to_csv(out_csv, index=False)
            wrote = True
            print(f"[feature-dataset] {split}: written (content changed)")
    else:
        np.savez_compressed(out_npz, X=X_feat, Y=Y,
                            features=np.array(FEATURE_NAMES),
                            targets=np.array(TARGET_NAMES),
                            horizons=np.array([6, 12, 24]))
        out_meta.to_csv(out_csv, index=False)
        wrote = True
        print(f"[feature-dataset] {split}: written")

    return {
        "split": split,
        "sequences": int(X_feat.shape[0]),
        "cyclones": int(out_meta["cyclone_id"].nunique()),
        "X_shape": list(X_feat.shape),
        "Y_shape": list(Y.shape),
        "dtype": str(X_feat.dtype),
        "nan": bool(np.isnan(X_feat).any() or np.isnan(Y).any()),
        "inf": bool(np.isinf(X_feat).any() or np.isinf(Y).any()),
        "wrote": wrote,
        "npz": str(out_npz),
        "metadata": str(out_csv),
    }


def build_feature_datasets(
    clean_dir: Path,
    chrono_dir: Path,
    quality_csv: Path,
    out_dir: Path,
) -> Dict[str, object]:
    """Build (or reuse) the three feature splits. Returns a summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "built_from": {
            "clean_dir": str(clean_dir),
            "chrono_dir": str(chrono_dir),
            "quality_csv": str(quality_csv),
            "policy": "Phase-2 CLEAN-only authoritative; no value modification",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_contract": {
            "X_in": "(N,5,7)", "X_out": "(N,5,16)",
            "feature_order_raw": FEATURE_NAMES_RAW,
            "derived_features": DERIVED_FEATURES,
            "target_order": TARGET_NAMES,
            "horizons_hours": [6, 12, 24],
        },
        "first_step_zero_fill_doc": FIRST_STEP_ZERO_FILL_DOC,
        "splits": {},
    }
    for split in SPLITS:
        summary["splits"][split] = build_one_split(
            split, clean_dir, chrono_dir, quality_csv, out_dir)

    cyc_ok, seq_ok = True, True
    for split in SPLITS:
        s = summary["splits"][split]
        exp = EXPECTED_CLEAN[split]
        cyc_ok &= bool(s["cyclones"] == exp["cyclones"])
        seq_ok &= bool(s["sequences"] == exp["sequences"])
    summary["cyclone_counts_match_phase2"] = cyc_ok
    summary["sequence_counts_match_phase2"] = seq_ok
    if not (cyc_ok and seq_ok):
        raise RuntimeError("feature dataset counts disagree with Phase-2 EXPECTED_CLEAN")
    return summary


def write_feature_contract(summary: Dict[str, object], path: Path) -> None:
    lines = [
        "# P4 Phase-4 FEATURE CONTRACT (LOCKED)",
        "",
        f"- Generated: {summary['created_at']}",
        f"- First-step policy: {summary['first_step_zero_fill_doc']}",
        "",
        "## Input contract",
        "",
        "| item | value |",
        "|---|---|",
        "| raw history | (N, 5, 7) float32 |",
        "| featured history | (N, 5, 16) float32 |",
        "| raw feature order | `" + ", ".join(summary["input_contract"]["feature_order_raw"]) + "` |",
        "| derived features | `" + ", ".join(summary["input_contract"]["derived_features"]) + "` |",
        "| target order | `" + ", ".join(summary["input_contract"]["target_order"]) + "` |",
        "| horizons | +6 / +12 / +24 hours |",
        "| SST units | degrees Celsius (untouched) |",
        "",
        f"## Dataset splits (from `phase2/results/canonical_chronological_clean` - CLEAN only)",
        "",
        "| split | sequences | cyclones | X shape | Y shape | nan/inf |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        s = summary["splits"][split]
        lines.append(
            f"| {split} | {s['sequences']} | {s['cyclones']} | "
            f"{tuple(s['X_shape'])} | {tuple(s['Y_shape'])} | {s['nan']}/{s['inf']} |")
    lines += [
        "",
        "## Causality",
        "",
        "- No target Y cell is ever used to construct an input feature.",
        "- No future history timestep is used.",
        "- No interpolation is introduced (the CLEAN source has none).",
        "- OHz delta_lon uses wrapped difference in (-180, 180]; no 0/360 jump.",
        "- movement_speed = Haversine(km) / 6 h expressed in km/h.",
        "- movement_direction: initial bearing clockwise from true north, [0,360).",
        "- environmental_wind_speed = sqrt(u^2+v^2) m/s; direction = meteorological "
        "FROM-convention, degrees clockwise from north, [0,360).",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


FEATURE_FORMULA_ROWS = [
    ("delta_lat", "deg", "lat(i) - lat(i-1); i=0 -> 0"),
    ("delta_lon", "deg", "wrapped (lon(i) - lon(i-1)) in (-180, 180]; i=0 -> 0"),
    ("movement_speed", "km/h", "Haversine(prev,cur) / 6h; i=0 -> 0"),
    ("movement_direction", "deg", "initial bearing from prev to cur, [0,360); i=0 -> 0"),
    ("wind_change", "km/h", "wind_speed(i) - wind_speed(i-1); i=0 -> 0"),
    ("pressure_change", "hPa", "pressure(i) - pressure(i-1); i=0 -> 0"),
    ("sst_change", "deg C", "sst(i) - sst(i-1); i=0 -> 0"),
    ("environmental_wind_speed", "m/s", "sqrt(wind_u^2 + wind_v^2) (no predecessor)"),
    ("environmental_wind_direction", "deg", "FROM-direction atan2(-u,-v) mod 360 (no predecessor)"),
]


def write_feature_engineering_report(summary: Dict[str, object], path: Path) -> None:
    lines = [
        "# P4 Phase-4 FEATURE ENGINEERING REPORT",
        "",
        f"- Generated: {summary['created_at']}",
        f"- Source dataset: `p4_forecasting/phase2/results/canonical_chronological_clean` "
        f"({summary['splits']['train']['sequences']} train / "
        f"{summary['splits']['val']['sequences']} val / "
        f"{summary['splits']['test']['sequences']} test sequences; "
        "CLEAN-only policy authoritative).",
        "",
        "## The 9 derived features",
        "",
        "| # | feature | units | formula / convention |",
        "|---|---|---|---|",
    ]
    for name, units, formula in FEATURE_FORMULA_ROWS:
        lines.append(f"| {name} | {units} | {formula} |")
    lines += [
        "",
        "## Longitude wrapping",
        "",
        "Canonical longitudes are stored in [0, 360). delta_lon uses the wrapped "
        "difference `(lon_i - lon_{i-1} + 180) mod 360 - 180`, so no artificial "
        "jump across the 0/360 boundary is created. Predicted longitudes are "
        "wrapped back into [0, 360) at inference.",
        "",
        "## First-step zero-fill policy",
        "",
        f"> {summary['first_step_zero_fill_doc']}",
        "",
        "## Causal rule",
        "",
        "At each history step i the engineered features depend only on steps "
        "<= i (for i=0 only on step 0). None of the nine features uses a future "
        "observation, the forecast targets, the i=-1 step, or any statistical "
        "summary of the whole window (no global mean/std enters feature values).",
        "",
        "## Why no future data is used",
        "",
        "A forecast issued at t_zero may legitimately only know information up to "
        "t_zero. Using t_zero+k or the +6/+12/+24 h targets would leak the answer "
        "into the input. The leakage tests (test_features.py, test_no_future_leak.py) "
        "mutate future steps and targets and assert the engineered features do not "
        "change.",
        "",
        "## Source dataset",
        "",
        f"- Clean dir: `{summary['built_from']['clean_dir']}`",
        f"- Chrono provenance: `{summary['built_from']['chrono_dir']}`",
        f"- Quality: `{summary['built_from']['quality_csv']}`",
        "",
        "## Resulting shapes",
        "",
        "| split | X | Y |",
        "|---|---|---|",
        "| train | (N,5,16) | (N,3,3) |",
        "| val | (N,5,16) | (N,3,3) |",
        "| test | (N,5,16) | (N,3,3) |",
        "",
        "## Quality filtering",
        "",
        "Only rows whose Phase-1 quality status == CLEAN are retained (Phase-2 "
        "authoritative policy). Counts equal the Phase-2 clean set exactly.",
        "",
        "## Missing-value handling",
        "",
        "The CLEAN source contains no NaN/Inf. No imputation is introduced. The "
        "only deterministic handling is the documented first-step zero-fill for "
        "predecessor-dependent features.",
        "",
        "## Verification results",
        "",
        "- Raw 7 columns byte-identical to the source (tested).",
        "- First-step predecessor-dependent features exactly zero (tested).",
        "- Future-step and target mutation do not alter engineered features (tested).",
        "- Longitude wrap and movement numerics validated (tested).",
        "- No NaN/Inf introduced (tested).",
        f"- Counts verified against Phase-2: {summary['cyclone_counts_match_phase2']} "
        f"(cyclones) / {summary['sequence_counts_match_phase2']} (sequences).",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_feature_dataset(dataset_dir: Path) -> Dict[str, object]:
    """Read the built feature dataset back into arrays (validation helper)."""
    out: Dict[str, object] = {}
    for split in SPLITS:
        z = np.load(dataset_dir / f"{split}.npz", allow_pickle=True)
        meta = pd.read_csv(dataset_dir / f"{split}_metadata.csv")
        out[split] = {
            "X": np.asarray(z["X"], np.float32),
            "Y": np.asarray(z["Y"], np.float32),
            "features": [str(x) for x in z["features"]],
            "targets": [str(x) for x in z["targets"]],
            "horizons": [int(h) for h in z["horizons"]],
            "meta": meta,
        }
    return out


def load_feature_splits(dataset_dir: Path) -> Dict[str, object]:
    return summarize_feature_dataset(dataset_dir)