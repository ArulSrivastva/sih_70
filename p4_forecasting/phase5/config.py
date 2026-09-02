"""Phase-5 constants and read-only path resolution.

Files are resolved against the *actual* Phase-4 landing zone so the champion,
its config and the training-only normalization statistics are always the ones
that were audited.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent          # .../p4_forecasting
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

PROJECT_ROOT = _PKG.parent
P4_FORECASTING = _PKG
PHASE4 = _PKG / "phase4"
PHASE5 = _PKG / "phase5"
PHASE2 = _PKG / "phase2"

# Canonical Phase-4 contract (kept in sync with phase4/common.py; the runtime
# code nevertheless re-reads feature_order from normalization_stats.json).
FEATURE_NAMES_RAW = [
    "lat", "lon", "wind_speed", "pressure", "sst", "wind_u", "wind_v",
]
DERIVED_FEATURES = [
    "delta_lat", "delta_lon", "movement_speed", "movement_direction",
    "wind_change", "pressure_change", "sst_change",
    "environmental_wind_speed", "environmental_wind_direction",
]
FEATURE_NAMES = FEATURE_NAMES_RAW + DERIVED_FEATURES          # 16
TARGET_NAMES = ["lat", "lon", "wind_speed"]

HISTORY_STEPS = 5
RAW_FEATURES = 7
N_FEATURES = 16
N_TARGETS = 3
SPACING_HOURS = 6
HISTORY_HOURS = 24
HORIZON_HOURS = [6, 12, 24]
HORIZON_TAGS = ["6h", "12h", "24h"]

DEFAULT_CHAMPION_ID = "EXP005"


def read_json(path: str | Path):
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def champion_experiment_id(phase4: Path = PHASE4) -> str:
    """Read the audited champion id from champion_model.json (no guessing)."""
    meta = phase4 / "results" / "champion_model.json"
    if meta.exists():
        return str(read_json(meta)["experiment_id"])
    return DEFAULT_CHAMPION_ID


@dataclass(frozen=True)
class Phase5Paths:
    """Resolved read-only inputs + the two Phase-5 write roots."""

    phase4: Path = PHASE4
    phase5: Path = PHASE5
    champion_id: str = DEFAULT_CHAMPION_ID

    @property
    def champion_dir(self) -> Path:
        return self.phase4 / "results" / "experiments" / self.champion_id

    @property
    def champion_checkpoint(self) -> Path:
        return self.champion_dir / "checkpoint.pt"

    @property
    def champion_config(self) -> Path:
        return self.champion_dir / "config.json"

    @property
    def normalization_stats(self) -> Path:
        return self.phase4 / "results" / "normalization_stats.json"

    @property
    def champion_meta(self) -> Path:
        return self.phase4 / "results" / "champion_model.json"

    @property
    def registry(self) -> Path:
        return self.phase4 / "results" / "experiment_registry.csv"

    @property
    def final_comparison(self) -> Path:
        return self.phase4 / "results" / "FINAL_COMPARISON.json"

    @property
    def feature_dataset(self) -> Path:
        return self.phase4 / "results" / "feature_dataset"

    @property
    def clean_data_dir(self) -> Path:
        return PHASE2 / "results" / "canonical_chronological_clean"

    @property
    def results_dir(self) -> Path:
        return self.phase5 / "results"

    @property
    def reports_dir(self) -> Path:
        return self.phase5 / "reports"

    def require(self) -> list:
        """Return the list of missing required read-only sources."""
        required = [
            self.champion_checkpoint, self.champion_config,
            self.normalization_stats, self.champion_meta, self.registry,
            self.final_comparison,
            self.phase4 / "features" / "feature_engineering.py",
            self.phase4 / "models" / "gru.py",
            self.phase4 / "training" / "normalization.py",
            self.phase4 / "inference" / "forecaster.py",
        ]
        return [str(p) for p in required if not p.exists()]


def default_paths(champion_id: str | None = None) -> Phase5Paths:
    if champion_id is None:
        champion_id = champion_experiment_id()
    return Phase5Paths(champion_id=champion_id)