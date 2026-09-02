"""Phase-5 source immutability snapshotting (read-only for consumed sources).

Snapshots every non-Phase-5 file under the project (excluding the huge
PS70-main.zip archive, bytecode and pytest caches) so we can prove later that
P1 / Phase-1 / Phase-2 / Phase-3 / Phase-4 and everything outside Phase-5 were
left unchanged.

Buckets (paths relative to cyclone-project/):

    p1        p4_forecasting/_source_p1, canonical, canonical_chrono
    phase1    p4_forecasting/canonical, canonical_chrono, audit, reports,
              scripts, logs
    phase2    p4_forecasting/phase2
    phase3    p4_forecasting/phase3
    phase4    p4_forecasting/phase4
    outside_phase5  every key (i.e. full tree outside p4_forecasting/phase5)
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
PROJECT = PKG.parent
RESULTS = PKG / "phase5" / "results"

SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc"}
SKIP_FILES = {"PS70-main.zip"}

BUCKETS = {
    "p1": (
        "p4_forecasting/_source_p1",
        "p4_forecasting/canonical",
        "p4_forecasting/canonical_chrono",
    ),
    "phase1": (
        "p4_forecasting/canonical",
        "p4_forecasting/canonical_chrono",
        "p4_forecasting/audit",
        "p4_forecasting/reports",
        "p4_forecasting/scripts",
        "p4_forecasting/logs",
    ),
    "phase2": ("p4_forecasting/phase2",),
    "phase3": ("p4_forecasting/phase3",),
    "phase4": ("p4_forecasting/phase4",),
}


def sha256(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_tree(root: Path, relative_to: Path,
                  skip_prefixes: tuple = ()) -> dict:
    """{relpath_posix: sha256} for every non-excluded file under root."""
    snap: dict = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix in SKIP_SUFFIXES or name in SKIP_FILES:
                continue
            rel = p.relative_to(relative_to).as_posix()
            if rel.startswith(skip_prefixes):
                continue
            snap[rel] = sha256(p)
    return snap


def master_snapshot() -> dict:
    """Full non-Phase5 tree snapshot keyed relative to cyclone-project/."""
    return snapshot_tree(
        PROJECT, PROJECT,
        skip_prefixes=("p4_forecasting/phase5/", "p4_forecasting/phase5"),
    )


def _keys_for(bucket_roots: tuple, master: dict) -> dict:
    return {k: v for k, v in master.items()
            if k.startswith(bucket_roots)}


def write_before(path: Path = RESULTS / "source_hashes_before.json") -> dict:
    """Right-size snapshot of every consumed source, taken before any Phase-5
    implementation step that could conceivably touch project data."""
    master = master_snapshot()
    payload = {
        "schema": "phase5 v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "policy": ("SHA256 of every non-Phase-5 file under the project "
                   "(excluding PS70-main.zip, __pycache__, *.pyc, "
                   ".pytest_cache and p4_forecasting/phase5 itself)."),
        "project_root": str(PROJECT),
        "excluded": ["PS70-main.zip", "__pycache__", "*.pyc",
                     ".pytest_cache", "p4_forecasting/phase5"],
        "bucket_roots": BUCKETS,
        "n_files": len(master),
        "hashes": master,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return payload


def read_before(path: Path = RESULTS / "source_hashes_before.json") -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: record the BEFORE snapshot first")
    return json.loads(path.read_text(encoding="utf-8"))


def changed_files(before: dict, after: dict):
    return sorted(k for k in set(before) | set(after)
                  if before.get(k) != after.get(k))


def bucket_changes(before_master: dict, after_master: dict) -> dict:
    """Per-bucket changed-file lists plus the outside_phase5 total."""
    out = {"outside_phase5": changed_files(before_master, after_master)}
    for bucket, roots in BUCKETS.items():
        b = _keys_for(roots, before_master)
        a = _keys_for(roots, after_master)
        out[bucket] = changed_files(b, a)
    return out


def write_after_report(after_master: dict, status: dict,
                       path: Path = RESULTS / "source_immutability_report.json",
                       note: str = "") -> dict:
    report = {
        "schema": "phase5 v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": status["status"],
        "fatal": status.get("fatal", False),
        "p1_changed": len(status["changes"]["p1"]),
        "phase1_changed": len(status["changes"]["phase1"]),
        "phase2_changed": len(status["changes"]["phase2"]),
        "phase3_changed": len(status["changes"]["phase3"]),
        "phase4_changed": len(status["changes"]["phase4"]),
        "outside_phase5_changed": len(status["changes"]["outside_phase5"]),
        "changed_files": status["changes"],
        "n_files_after": len(after_master),
        "note": note or "No Phase-1/2/3/4 or outside-Phase-5 file changed.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return report


def make_status(before_master: dict, after_master: dict) -> dict:
    changes = bucket_changes(before_master, after_master)
    failed = [k for k, v in changes.items() if v]
    return {
        "status": "PASS" if not failed else "FAIL",
        "fatal": bool(failed),
        "changes": changes,
    }


def verify_immutability(before_payload: dict, after_master: dict,
                        note: str = "") -> dict:
    status = make_status(before_payload["hashes"], after_master)
    return write_after_report(after_master, status, note=note)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase-5 source snapshot")
    ap.add_argument("--before", action="store_true",
                    help="write source_hashes_before.json (run before impl)")
    ap.add_argument("--check", action="store_true",
                    help="compare current tree against stored BEFORE snapshot")
    args = ap.parse_args()
    if args.before:
        p = write_before()
        print(f"[snapshot] BEFORE recorded: {p['n_files']} files -> "
              f"{RESULTS / 'source_hashes_before.json'}")
    else:
        print("use --before (record) or --check (verify)")