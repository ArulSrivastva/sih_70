"""Shared constants, write guard and hashing helpers for P4 Phase 4.

Everything Phase 4 produces must live under ``p4_forecasting/phase4``.  The
``WriteGuard`` enforces this at the point every output path is created.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

P4_DIR = Path(__file__).resolve().parent                 # .../p4_forecasting/phase4
PKG_DIR = P4_DIR.parent                                  # .../p4_forecasting
PROJECT_ROOT = PKG_DIR.parent                            # cyclone-project/

# Canonical Phase-4 dataset contract (locked with the approved Phase-4 plan).
FEATURE_NAMES_RAW = [
    "lat", "lon", "wind_speed", "pressure", "sst", "wind_u", "wind_v",
]
DERIVED_FEATURES = [
    "delta_lat", "delta_lon", "movement_speed", "movement_direction",
    "wind_change", "pressure_change", "sst_change",
    "environmental_wind_speed", "environmental_wind_direction",
]
FEATURE_NAMES = FEATURE_NAMES_RAW + DERIVED_FEATURES      # 16 columns
TARGET_NAMES = ["lat", "lon", "wind_speed"]
HORIZON_HOURS = [6, 12, 24]
HISTORY_STEPS = 5
N_FEATURES = 16
N_TARGETS = 3
N_HORIZONS = 3
HORIZONS = ["6h", "12h", "24h"]

# Expected canonical_chrono SHA256 prefixes recorded at the Phase-1/2 handoff.
EXPECTED_CHRONO_SHA256_PREFIX = {
    "train.npz": "df70303e",
    "val.npz": "48cf065d",
    "test.npz": "89e9c2e2",
}

# Expected CLEAN-only counts (Phase-2 authoritative CLEAN policy).
EXPECTED_CLEAN = {
    "train": {"sequences": 1212, "cyclones": 57},
    "val": {"sequences": 231, "cyclones": 13},
    "test": {"sequences": 198, "cyclones": 10},
}


class WriteGuard:
    """Reject any write whose resolved path escapes the phase4 root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def check(self, path: str | Path) -> Path:
        p = Path(path).absolute()
        try:
            in_root = p.is_relative_to(self.root)
        except AttributeError:  # pragma: no cover - older pythons
            in_root = False
        if not in_root:
            raise PermissionError(
                f"write guard: refusing path outside {self.root}: {p}")
        return p

    def join(self, *parts: str) -> Path:
        return self.check(self.root.joinpath(*parts))


def sha256(path: str | Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _iter_snapshot_files(base: Path) -> List[Path]:
    files: List[Path] = []
    for root, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
        for name in sorted(names):
            p = Path(root) / name
            if p.suffix == ".pyc":
                continue
            files.append(p)
    return sorted(files)


def snapshot_dir(base: str | Path, relative_to: str | Path | None = None) -> Dict[str, str]:
    """SHA256 of every file under base. Returns {relative_path: sha256}."""
    base = Path(base)
    rel = Path(relative_to) if relative_to is not None else base
    snap: Dict[str, str] = {}
    if base.exists():
        for p in _iter_snapshot_files(base):
            snap[p.relative_to(rel).as_posix()] = sha256(p)
    return snap


def files_changed(before: Mapping[str, str], after: Mapping[str, str]) -> List[str]:
    return sorted(k for k in before if before[k] != after.get(k))


def diff_snapshots(before: Mapping[str, str], after: Mapping[str, str]) -> Dict[str, List[str]]:
    """Return keys that changed/added/removed between two snapshots."""
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    added = sorted(k for k in after if k not in before)
    removed = sorted(k for k in before if k not in after)
    return {"changed": changed, "added": added, "removed": removed}


def require(*paths: Iterable[Path]) -> List[Path]:
    missing = [str(p) for p in paths for p in [Path(p)] if not p.exists()]
    if missing:
        raise FileNotFoundError("missing required source files:\n  " + "\n  ".join(missing))
    return [Path(p) for p in paths]