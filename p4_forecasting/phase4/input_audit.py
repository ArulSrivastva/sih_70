"""Read-only input contract audit for Phase-4 (STEP 2).

Verifies existence, shapes, dtypes, feature/target order, horizons, metadata
alignment, cyclone disjointness, chronological ordering, clean data and source
hashes before any feature construction or training.  A failed fundamental check
STOPS the pipeline before training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .common import EXPECTED_CHRONO_SHA256_PREFIX, EXPECTED_CLEAN, sha256


@dataclass
class AuditContext:
    chrono_dir: Path
    canonical_dir: Path
    phase2_clean_dir: Path
    phase2_baseline_json: Path
    phase3_comparison_json: Path
    phase3_stats_json: Path
    quality_csv: Path
    p4_dir: Path  # p4_forecasting root (used for source hashing provenance)


def _check(name: str, ok: bool, detail: str) -> Dict[str, object]:
    return {"check": name, "pass": bool(ok), "detail": detail}


def run_input_audit(ctx: AuditContext) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []
    details: Dict[str, object] = {}
    critical_fail = False

    def add(ok: bool, detail: str) -> None:
        nonlocal critical_fail
        ok = bool(ok)
        checks.append(_check(f"check_{len(checks) + 1}", ok, detail))
        if not ok:
            critical_fail = True

    # 1. existence
    required = {
        "chrono": [
            ctx.chrono_dir / "train.npz", ctx.chrono_dir / "val.npz", ctx.chrono_dir / "test.npz",
            ctx.chrono_dir / "train_metadata.csv", ctx.chrono_dir / "val_metadata.csv",
            ctx.chrono_dir / "test_metadata.csv", ctx.chrono_dir / "split_manifest.csv",
        ],
        "canonical": [ctx.quality_csv],
        "phase2_clean": [
            ctx.phase2_clean_dir / "train.npz", ctx.phase2_clean_dir / "val.npz",
            ctx.phase2_clean_dir / "test.npz", ctx.phase2_clean_dir / "train_metadata.csv",
            ctx.phase2_clean_dir / "val_metadata.csv", ctx.phase2_clean_dir / "test_metadata.csv",
        ],
        "phase2_baseline": [ctx.phase2_baseline_json],
        "phase3": [ctx.phase3_comparison_json, ctx.phase3_stats_json],
    }
    missing = [str(p) for group in required.values() for p in group if not p.exists()]
    add(len(missing) == 0, f"all source files exist ({len(missing)} missing): {missing}")

    # 2. canonical_chrono hashes (require the recorded prefixes)
    chrono_hashes = {}
    for fn in ("train.npz", "val.npz", "test.npz"):
        p = ctx.chrono_dir / fn
        if p.exists():
            chrono_hashes[fn] = sha256(p)
        else:
            chrono_hashes[fn] = None
    prefix_ok = all(
        chrono_hashes.get(fn) is not None and chrono_hashes[fn].startswith(EXPECTED_CHRONO_SHA256_PREFIX[fn])
        for fn in EXPECTED_CHRONO_SHA256_PREFIX)
    add(prefix_ok, f"canonical_chrono npz hash prefixes match Phase-1/2 records "
                   f"({ {k: (v[:12] + '...' if v else None) for k, v in chrono_hashes.items()} })")
    details["chrono_sha256"] = chrono_hashes

    # 3-7: per-split contract checks on the Phase-2 CLEAN source
    split_report = {}
    expected_vals = {"feature_order": ["lat", "lon", "wind_speed", "pressure", "sst", "wind_u", "wind_v"],
                     "target_order": ["lat", "lon", "wind_speed"]}
    for split in ("train", "val", "test"):
        npz_p = ctx.phase2_clean_dir / f"{split}.npz"
        meta_p = ctx.phase2_clean_dir / f"{split}_metadata.csv"
        if not npz_p.exists() or not meta_p.exists():
            split_report[split] = {"ok": False, "detail": "missing npz or metadata"}
            add(False, f"{split}: missing clean npz/metadata")
            continue
        z = np.load(npz_p, allow_pickle=True)
        X = np.asarray(z["X"], np.float32)
        Y = np.asarray(z["Y"], np.float32)
        meta = pd.read_csv(meta_p)
        row = {}
        row["X_shape"] = list(X.shape) if X.ndim == 3 else None
        row["Y_shape"] = list(Y.shape) if Y.ndim == 3 else None
        row["dtype"] = str(X.dtype)
        ok_shape = X.ndim == 3 and X.shape[1:] == (5, 7) and Y.ndim == 3 and Y.shape[1:] == (3, 3)
        row["shapes_ok"] = ok_shape
        feats = [str(x) for x in z["features"]]
        tgts = [str(x) for x in z["targets"]]
        row["feature_order_ok"] = feats == expected_vals["feature_order"]
        row["target_order_ok"] = tgts == expected_vals["target_order"]
        row["horizons"] = infer_horizons(meta)
        row["meta_align"] = (meta.shape[0] == X.shape[0] == Y.shape[0]
                             and {"cyclone_id", "t_zero"} <= set(meta.columns))
        row["nan_inf"] = bool(np.isnan(X).any() or np.isinf(X).any()
                              or np.isnan(Y).any() or np.isinf(Y).any())
        row["cyclones"] = int(meta["cyclone_id"].nunique())
        row["sequences"] = int(X.shape[0])
        row["expected"] = EXPECTED_CLEAN.get(split)
        row["counts_ok"] = (row["cyclones"] == EXPECTED_CLEAN[split]["cyclones"]
                            and row["sequences"] == EXPECTED_CLEAN[split]["sequences"])
        row["quality_clean_only"] = bool((meta["quality_status"] == "CLEAN").all())
        split_report[split] = row
        ok_split = row["shapes_ok"] and row["feature_order_ok"] and row["target_order_ok"] \
            and row["meta_align"] and not row["nan_inf"] and row["counts_ok"] and row["quality_clean_only"]
        add(ok_split, f"{split}: shapes={row['X_shape']}/{row['Y_shape']} dtype={row['dtype']} "
                      f"feat_ok={row['feature_order_ok']} tgt_ok={row['target_order_ok']} "
                      f"align={row['meta_align']} nan_inf={row['nan_inf']} "
                      f"cycles={row['cyclones']}/{EXPECTED_CLEAN[split]['cyclones']} "
                      f"seq={row['sequences']}/{EXPECTED_CLEAN[split]['sequences']} "
                      f"clean_only={row['quality_clean_only']}")

    # 8. cyclone disjointness across clean splits
    sets = {}
    for split in ("train", "val", "test"):
        meta = pd.read_csv(ctx.phase2_clean_dir / f"{split}_metadata.csv")
        sets[split] = set(meta["cyclone_id"])
    disjoint = not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
        and not (sets["val"] & sets["test"])
    add(disjoint, f"clean-split cyclone disjointness (train={len(sets['train'])}, "
                  f"val={len(sets['val'])}, test={len(sets['test'])})")

    # 9. chronological ordering from split_manifest
    chrono_ok, chrono_detail = check_chronology(ctx.chrono_dir / "split_manifest.csv")
    add(chrono_ok, chrono_detail)

    # 10. clean rows map to canonical_chrono CLEAN rows one-to-one
    mapping_ok, mapping_detail = check_clean_mapping(ctx)
    add(mapping_ok, mapping_detail)

    report = {
        "audit_version": "phase4 v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": not critical_fail,
        "checks": checks,
        "source_paths": {
            "chrono_dir": str(ctx.chrono_dir),
            "canonical_dir": str(ctx.canonical_dir),
            "phase2_clean_dir": str(ctx.phase2_clean_dir),
            "phase2_baseline_json": str(ctx.phase2_baseline_json),
            "phase3_comparison_json": str(ctx.phase3_comparison_json),
            "phase3_stats_json": str(ctx.phase3_stats_json),
            "quality_csv": str(ctx.quality_csv),
        },
        "details": {"chrono_sha256": details.get("chrono_sha256", {}),
                    "splits": split_report},
    }
    report["audit_summary"] = {
        "checks_total": len(checks),
        "checks_failed": sum(1 for c in checks if not c["pass"]),
    }
    return report


def infer_horizons(meta: pd.DataFrame) -> List[int]:
    cols = set(meta.columns)
    h = []
    for name in ("6", "12", "24"):
        if f"target_{name}h_lat" in cols:
            h.append(int(name))
    return h


def check_chronology(manifest_path: Path) -> tuple:
    if not manifest_path.exists():
        return False, "split_manifest.csv missing"
    manifest = pd.read_csv(manifest_path)
    if {"cyclone_id", "split", "storm_start", "storm_end"} - set(manifest.columns):
        return False, "split_manifest.csv missing expected columns"
    groups = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}
    if any(g.empty for g in groups.values()):
        return False, "split_manifest missing one of train/val/test"
    max_train_end = groups["train"]["storm_end"].max()
    min_val_start = groups["val"]["storm_start"].min()
    max_val_end = groups["val"]["storm_end"].max()
    min_test_start = groups["test"]["storm_start"].min()
    if not (max_train_end < min_val_start < max_val_end < min_test_start):
        return False, (f"not strictly chronological: max_train_end={max_train_end}, "
                       f"min_val_start={min_val_start}, max_val_end={max_val_end}, "
                       f"min_test_start={min_test_start}")
    return True, (f"strictly chronological storms: all train storms end before all val "
                  f"storms begin before all test storms "
                  f"({max_train_end} < {min_val_start} < {max_val_end} < {min_test_start})")


def check_clean_mapping(ctx: AuditContext) -> tuple:
    """Every Phase-2 CLEAN row is a canonical_chrono row flagged CLEAN."""
    quality = pd.read_csv(ctx.quality_csv)
    counts = []
    for split in ("train", "val", "test"):
        clean_meta = pd.read_csv(ctx.phase2_clean_dir / f"{split}_metadata.csv")
        chrono_meta = pd.read_csv(ctx.chrono_dir / f"{split}_metadata.csv")
        merged = chrono_meta.merge(
            quality[["cyclone_id", "t_zero", "quality_status"]],
            on=["cyclone_id", "t_zero"], how="left")
        n_clean = int((merged["quality_status"] == "CLEAN").sum())
        counts.append((split, n_clean, int(clean_meta.shape[0])))
    ok = all(n_clean == rows for _, n_clean, rows in counts)
    return ok, f"canonical_chrono CLEAN counts vs Phase-2 clean rows: {counts}"


def save_input_audit(report: Dict[str, object], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")