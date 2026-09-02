"""integration_api / config.py

Constants and read-only location metadata for the integration layer.

Everything scientific stays inside the audited artifacts:
  * P4 forecast        -> phase5 / phase6 (already audited)
  * P2 detection model -> zip bytes of PS70-main/models/detection/model_weights.pt
  * P3 classification  -> zip bytes of PS70-main/models/classification/*.pt/.pkl
  * reference INSAT-3D IR frame -> a deterministic sample from the zip, never
    claimed to be a live feed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

_PKG = Path(__file__).resolve().parent.parent          # .../p4_forecasting
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

PROJECT_ROOT = _PKG.parent
P4_FORECASTING = _PKG
INTEGRATION_API = _PKG / "integration_api"

# ---------------------------------------------------------------------------
# The one artifact bundle everything P2/P3 reads from.  Never extracted.
# ---------------------------------------------------------------------------
ZIP_PATH = PROJECT_ROOT / "PS70-main.zip"

P2_WEIGHTS_REL = "PS70-main/models/detection/model_weights.pt"
P2_CANDIDATE_WEIGHTS_PATH = PROJECT_ROOT / "models/detection/model_weights_phase9_E9_2.pt"
E9_2_PATTERNS: List[str] = ["curved_band", "eye_visible", "shear_pattern"]
E9_2_CATEGORIES: List[str] = ["Cyclonic Storm", "Deep Depression", "Depression", "Very Severe Cyclonic Storm"]

P3_IMAGE_WEIGHTS_REL = "PS70-main/models/classification/image_only_model.pt"
P3_TABULAR_MODEL_REL = "PS70-main/models/classification/tabular_multisource_model.pkl"

# Reference INSAT-3D IR frame used as the "current pass" for P2/P3 inference.
# Picked deterministically; the choice is disclosed in every response's
# provenance block and in meta.source.  This is a sample frame, NOT a live feed.
INSAT_FINAL_DIR = "PS70-main/data/raw/insat/insat3d_raw_cyclone_ds/CYCLONE_DATASET_FINAL/"
INSAT_FINAL_DIR_2 = "PS70-main/data/raw/insat_kaggle/insat3d_raw_cyclone_ds/CYCLONE_DATASET_FINAL/"
P3_IMAGE_DIR = "PS70-main/data/processed/classification/image_only_kaggle/images/"
DEFAULT_REFERENCE_IMAGES = (
    INSAT_FINAL_DIR + "45(1).jpg",
    INSAT_FINAL_DIR_2 + "45(1).jpg",
    INSAT_FINAL_DIR + "45.jpg",
)

# ---------------------------------------------------------------------------
# IMD intensity classes as defined by the audited Phase-3 classifier.
# ---------------------------------------------------------------------------
IMD_CLASSES: List[str] = [
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm",
]

IMD_SHORT_CODES: Dict[str, str] = {
    "Depression": "D",
    "Deep Depression": "DD",
    "Cyclonic Storm": "CS",
    "Severe Cyclonic Storm": "SCS",
    "Very Severe Cyclonic Storm": "VSCS",
    "Extremely Severe Cyclonic Storm": "ESCS",
    "Super Cyclonic Storm": "SuCS",
}

# ---------------------------------------------------------------------------
# Replacement meta when the request does not carry one (demo ANIKA narrative).
# ---------------------------------------------------------------------------
DEFAULT_META = {
    "systemId": "BOB07",
    "systemName": "Cyclonic Storm ANIKA",
    "basin": "Bay of Bengal",
    "lastPass": "2026-08-26T05:30:00Z",
    "source": "INSAT-3D IR · reference frame from PS70-main.zip",
}

# ---------------------------------------------------------------------------
# Landfall / risk heuristics: never model outputs, always disclosed as
# deterministic server-side approximations in the provenance block.
# ---------------------------------------------------------------------------
LANDFALL_THRESHOLD_KM = 200.0
RISK_WIND_REF_KMH = 30.0

# Compact set of North-Indian-Ocean coastal reference points (lat, lon deg)
# used only to estimate coastline proximity for the landfall heuristic.
NIO_COAST_POINTS: List[Tuple[float, float]] = [
    (8.07, 77.50), (8.80, 78.10), (9.30, 79.30), (10.30, 79.86),
    (10.70, 79.80), (11.70, 79.77), (12.00, 79.85), (13.08, 80.27),
    (13.40, 80.30), (14.20, 80.10), (15.20, 80.05), (16.40, 81.45),
    (17.00, 82.20), (17.68, 83.28), (18.10, 83.65), (18.90, 84.10),
    (19.30, 84.80), (19.90, 86.10), (20.28, 86.70), (20.95, 86.95),
    (21.15, 86.75), (21.49, 87.00), (21.60, 87.40), (21.70, 88.00),
    (21.85, 88.30), (21.90, 89.20), (21.80, 90.10), (22.00, 90.80),
    (22.20, 91.10), (22.35, 91.80), (22.40, 92.10), (16.50, 94.30),
    (18.50, 94.30), (20.50, 93.00), (9.70, 80.90), (8.60, 81.20),
    (8.20, 80.05), (8.07, 77.55), (9.97, 76.27), (10.20, 76.20),
    (11.30, 75.80), (12.90, 74.80), (13.90, 74.60), (14.80, 74.10),
    (15.70, 73.70), (16.70, 73.30), (17.70, 73.10), (18.50, 73.00),
    (19.00, 72.80), (19.80, 72.75), (20.90, 70.40), (21.20, 72.70),
    (21.60, 69.60), (22.20, 68.96), (22.80, 69.60), (23.20, 68.90),
]

# Forecast horizons exposed by the audited Phase-4 champion (EXP005).
FORECAST_HORIZONS: List[int] = [6, 12, 24]

# ---------------------------------------------------------------------------
# North Indian Ocean policy domain (upstream guard).
#
# The IMD/RSMC New Delhi mandate and the audited Phase-1 IBTrACS NI-basin
# extraction both live within this authoritative box.  The phase-6 layer only
# enforces global physical bounds (lat in [-90, 90], lon in [0, 360)), so an
# observation can pass phase-6 yet be geophysically outside the North Indian
# Ocean (e.g. lat < 0 pushes into the South Indian / southern Atlantic or
# Pacific basins the IMD does not forecast).  This integration layer therefore
# guards the NIO box UPSTREAM of phase-6, rejecting non-NIO input outright
# (never silently clamped) with error code OUT_OF_DOMAIN.
#
# Longitude uses the project's canonical degrees-East [0, 360) convention, so
# the NIO box is 40 deg-E .. 100 deg-E.
# ---------------------------------------------------------------------------
NIO_LAT_RANGE: Tuple[float, float] = (0.0, 30.0)
NIO_LON_RANGE: Tuple[float, float] = (40.0, 100.0)
OUT_OF_DOMAIN = "OUT_OF_DOMAIN"