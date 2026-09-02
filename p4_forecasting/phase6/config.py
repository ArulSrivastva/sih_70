"""Phase-6 constants and read-only path/service metadata.

Everything read here points back at the audited Phase-4/Phase-5 artifacts.
No scientific value is re-derived or hard-coded where an artifact exists.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

_PKG = Path(__file__).resolve().parent.parent          # .../p4_forecasting
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

PROJECT_ROOT = _PKG.parent
P4_FORECASTING = _PKG
PHASE6 = _PKG / "phase6"

from phase5.config import (HISTORY_HOURS, HISTORY_STEPS, N_FEATURES,
                           default_paths, read_json, champion_experiment_id)  # noqa: E402

API_NAME = "cyclone-forecasting"
API_PHASE = "phase6"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Local frontend development only.  Never open the API to the world.
CORS_ORIGINS = [
    "http://localhost:3000",   # Create React App / Next dev
    "http://localhost:5173",   # Vite dev server
]
CORS_ALLOW_CREDENTIALS = False

MAX_REQUEST_BYTES = 256 * 1024      # request-size guard (soft sanity bound)
MAX_HISTORY_STEPS = 5               # the API accepts exactly 5 observations
REQUIRED_OBSERVATION_FIELDS = [
    "timestamp", "latitude", "longitude", "wind_speed_kmh", "pressure_hpa",
    "sst", "wind_u", "wind_v",
]

# Physical bounds (longitude: canonical North-Indian-Ocean degrees East 0..360)
LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (0.0, 360.0)
WIND_RANGE = (0.0, 400.0)
PRESSURE_RANGE = (850.0, 1100.0)
SST_RANGE = (-5.0, 45.0)

SPACING_HOURS = 6

ERROR_CODES = {
    "INVALID_REQUEST",
    "INVALID_HISTORY_LENGTH",
    "INVALID_TIMESTAMP",
    "INVALID_HISTORY_SPACING",
    "NON_MONOTONIC_HISTORY",
    "INVALID_LATITUDE",
    "INVALID_LONGITUDE",
    "INVALID_WIND",
    "MISSING_FEATURE",
    "NON_FINITE_VALUE",
    "MODEL_NOT_READY",
    "INFERENCE_ERROR",
}

# Phase-5 (service) internal codes -> phase-6 public codes.
# Fine-grained range validation happens in phase6/schemas BEFORE the service
# is called, so the out_of_range fallback is a safety net only.
PHASE5_TO_PHASE6_CODE = {
    "missing_input": "INVALID_REQUEST",
    "wrong_shape": "INVALID_HISTORY_LENGTH",
    "bad_input": "INVALID_REQUEST",
    "insufficient_history": "INVALID_HISTORY_LENGTH",
    "nan_values": "NON_FINITE_VALUE",
    "infinite_values": "NON_FINITE_VALUE",
    "out_of_range": "INVALID_REQUEST",
    "malformed_timestamp": "INVALID_TIMESTAMP",
    "non_monotonic_timestamps": "NON_MONOTONIC_HISTORY",
    "timestamps_not_6hourly": "INVALID_HISTORY_SPACING",
    "internal_error": "INFERENCE_ERROR",
}


@dataclass(frozen=True)
class Phase6Paths:
    """Phase-5 (read-only) + Phase-6 (write) roots and output locations."""

    phase5: 'Phase5Paths'
    results_dir: Path
    reports_dir: Path
    examples_dir: Path
    tests_dir: Path


def phase6_paths() -> Phase6Paths:
    p5 = default_paths()
    return Phase6Paths(
        phase5=p5,
        results_dir=PHASE6 / "results",
        reports_dir=PHASE6 / "reports",
        examples_dir=PHASE6 / "examples",
        tests_dir=PHASE6 / "tests",
    )


def champion_identity() -> Dict[str, object]:
    """Read experiment metadata from the audited Phase-4 champion config +
    the Phase-4 selection record (never hard-coded scientific claims)."""
    paths = default_paths()
    cfg = read_json(paths.champion_config)
    meta = read_json(paths.champion_meta)
    stats = read_json(paths.normalization_stats)
    return {
        "experiment_id": str(meta["experiment_id"]),
        "model": str(cfg.get("model", "")),
        "loss": str(cfg.get("loss", "")),
        "hidden_size": int(cfg.get("hidden_size")),
        "layers": int(cfg.get("layers")),
        "input_size": int(cfg.get("input_size")),
        "output_size": int(cfg.get("output_size")),
        "history_steps": int(HISTORY_STEPS),
        "history_hours": int(HISTORY_HOURS),
        "feature_count": len(stats.get("feature_order", [])),
        "horizons": [int(h) for h in cfg.get("horizons_hours", [])],
        "targets": ["lat", "lon", "wind_speed_kmh"],
        "validation_primary_score": meta.get("primary_score"),
    }