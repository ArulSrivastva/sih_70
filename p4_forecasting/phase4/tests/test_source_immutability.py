"""Source-immutability test: Phase-1/P1/Phase-2/Phase-3 artifacts must be
unchanged by every Phase-4 run (self-healing baseline management)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phase4.common import PKG_DIR, diff_snapshots, sha256
from phase4.immutability import (
    PHASE1_DIRS,
    PHASE2_DIRS,
    PHASE3_DIRS,
    P1_DIRS,
    compute_immutability_report,
    group_snapshots,
)

RESULTS_DIR = PKG_DIR / "phase4" / "results"


def test_bucket_directory_sets_cover_everything_but_phase4():
    covered = set(P1_DIRS + PHASE1_DIRS + PHASE2_DIRS + PHASE3_DIRS)
    top_level = {p.name for p in PKG_DIR.iterdir() if p.is_dir() and p.name != "phase4"}
    assert covered == top_level, f"uncovered dirs: {top_level - covered}"


def test_snapshot_deterministic():
    a = group_snapshots(PKG_DIR)
    b = group_snapshots(PKG_DIR)
    for name in a:
        assert a[name] == b[name]


def test_sha_reference():
    tmp = Path(PKG_DIR / "phase4" / "tests" / ".scratch")
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "ref.txt"
    p.write_text("immutability reference", encoding="utf-8")
    import hashlib
    assert sha256(p) == hashlib.sha256(b"immutability reference").hexdigest()


def test_write_from_baseline_is_self_healing_or_pass():
    """If a baseline report exists, current state must match it bucket-wise;
       otherwise a fresh baseline (= current) is created so future runs can
       detect any modification."""
    report_path = RESULTS_DIR / "source_immutability_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        after = group_snapshots(PKG_DIR)
        changed = {}
        for name in ("p1", "phase1", "phase2", "phase3", "files_outside_phase4"):
            before = report["baseline"][name]
            d = diff_snapshots(before, after[name])
            changed[name] = d["changed"] + d["added"] + d["removed"]
            assert not changed[name], f"immutability broke in bucket {name}: {changed[name]}"
    else:
        before = group_snapshots(PKG_DIR)
        report = compute_immutability_report(before, before)
        assert report["verdict"] == "PASS"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def test_no_phase4_output_left_outside_phase4_dir():
    """Nothing Phase-4 writes may sit outside p4_forecasting/phase4."""
    for name in ("p4_raw_output", "phase4_outside_leak"):
        candidate = PKG_DIR / name
        assert not candidate.exists(), f"illegal Phase-4 output location: {candidate}"