"""Source immutability for Phase-4 (hashes of every consumed Read-only artifact).

All Phase-1/P1/Phase-2/Phase-3 artifacts are hashed BEFORE and AFTER every
Phase-4 run.  Any change outside phase4/ is a FAIL.  The report also supports
the standalone ``test_source_immutability.py`` (self-healing baseline).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .common import PKG_DIR, snapshot_dir, diff_snapshots

P1_DIRS = ["_source_p1", "canonical", "canonical_chrono"]
PHASE1_DIRS = ["canonical", "canonical_chrono", "audit", "reports", "scripts", "logs"]
PHASE2_DIRS = ["phase2"]
PHASE3_DIRS = ["phase3"]
ALL_OUTSIDE_DIRS = sorted(
    set(P1_DIRS + PHASE1_DIRS + PHASE2_DIRS + PHASE3_DIRS))

# Extra immutable top-level project files relevant to the P1 deliverable that
# Phases 2/3 also treat as read-only (hash-only provenance, relative to repo
# root).  NOTE: PS70-main.zip is deliberately excluded from hashing here - it is
# a 1.18 GB P1 delivery artifact already inventory-hashed at Phase-1.
EXTRA_PROJECT_FILES = [
    "AUDIT_P1_DELIVERY.md",
    "DATASET_REPORT.md",
]


def _group_snapshot(pkg: Path, dirs: List[str]) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for name in dirs:
        snap.update(snapshot_dir(pkg / name, relative_to=pkg))
    return snap


def _project_extra_snapshot() -> Dict[str, str]:
    from .common import PROJECT_ROOT
    out = {}
    for fn in EXTRA_PROJECT_FILES:
        p = PROJECT_ROOT / fn
        if p.exists():
            out[fn] = snapshot_dir(p, relative_to=PROJECT_ROOT).get(fn, _sha(p))
    return out


def _sha(path: Path) -> str:
    from .common import sha256
    return sha256(path)


def group_snapshots(pkg: Path = PKG_DIR) -> Dict[str, Dict[str, str]]:
    """Full current snapshots for the p1/phase1/phase2/phase3/outside buckets."""
    pkg = Path(pkg)
    outside = {}
    for name in ALL_OUTSIDE_DIRS:
        outside.update(snapshot_dir(pkg / name, relative_to=pkg))
    return {
        "p1": _group_snapshot(pkg, P1_DIRS),
        "phase1": _group_snapshot(pkg, PHASE1_DIRS),
        "phase2": _group_snapshot(pkg, PHASE2_DIRS),
        "phase3": _group_snapshot(pkg, PHASE3_DIRS),
        "project_extra": _project_extra_snapshot(),
        "files_outside_phase4": outside,
    }


def compute_immutability_report(
    before: Dict[str, Dict[str, str]],
    after: Dict[str, Dict[str, str]],
    recorded_at: str | None = None,
) -> Dict[str, object]:
    counts = {}
    changed = {}
    for name in ("p1", "phase1", "phase2", "phase3", "files_outside_phase4"):
        d = diff_snapshots(before[name], after[name])
        lst = d["changed"] + d["added"] + d["removed"]
        lst = sorted(set(lst))
        changed[name] = lst
        counts[f"{name}_modified"] = len(lst)
    counts["outside_phase4_modified"] = counts["files_outside_phase4_modified"]
    verdict = "PASS" if sum(counts[f"{k}_modified"] for k in
                            ("p1", "phase1", "phase2", "phase3", "files_outside_phase4")) == 0 else "FAIL"
    return {
        "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(),
        "groups": {
            "p1": {"dirs": P1_DIRS},
            "phase1": {"dirs": PHASE1_DIRS},
            "phase2": {"dirs": PHASE2_DIRS},
            "phase3": {"dirs": PHASE3_DIRS},
            "files_outside_phase4": {"dirs": ["every top-level dir under p4_forecasting except phase4"]},
        },
        "counts": counts,
        "verdict": verdict,
        "changed": changed,
        "baseline": before,
        "note": (
            "baseline = SHA256 snapshot taken before the Phase-4 run/step; any "
            "file changed/added/removed outside p4_forecasting/phase4 is a FAIL."),
    }


def write_immutability_report(report: Dict[str, object], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")