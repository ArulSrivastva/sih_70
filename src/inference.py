"""src / inference.py

Standalone deterministic Python inference engine for the cyclone detection,
classification, and forecasting pipeline.

Usage:
    from src.inference import run_inference
    res = run_inference("data/p2_phase7_genuine/crops/p7_mosdac_FANI_2019_0000_Very_Severe_Cyclonic_Storm.png")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
P4_DIR = ROOT / "p4_forecasting"
if str(P4_DIR) not in sys.path:
    sys.path.insert(0, str(P4_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration_api.p2_detector import get_detector
from integration_api.p3_classifier import get_image_classifier, get_tabular_classifier, tabular_status
from integration_api.analyzer import CycloneAnalyzer
from phase6.integration.forecasting_adapter import ForecastingAdapter
from integration_api.schemas import AnalyzeRequest
from phase6.schemas.requests import ForecastObservation


def run_inference(image_input: Union[str, Path, Image.Image],
                  history_input: Optional[Union[str, Path, List[Dict[str, Any]]]] = None,
                  meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute end-to-end inference across P2, P3, and P4 models."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert("RGB")
        img_name = Path(image_input).name
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        img_name = "in_memory_image.png"
    else:
        raise ValueError(f"Unsupported image_input type: {type(image_input)}")

    # 1. P2 Detection & Pattern Recognition
    det = get_detector().detect(img)

    # 2. P3 Classification
    tab_stat = tabular_status()
    history_obs = None
    if history_input is not None:
        if isinstance(history_input, (str, Path)):
            with open(history_input, "r", encoding="utf-8") as f:
                raw_hist = json.load(f)
            history_data = raw_hist.get("history", raw_hist)
        else:
            history_data = history_input
        history_obs = [ForecastObservation(**h) if isinstance(h, dict) else h for h in history_data]

    if history_obs and len(history_obs) >= 5 and tab_stat["available"]:
        last = history_obs[-1]
        tab_res = get_tabular_classifier().classify_tabular(
            last.latitude, last.longitude, last.sst,
            last.pressure_hpa, last.wind_u, last.wind_v)
        classification = {
            "category": tab_res["category"],
            "confidence": tab_res["confidence"],
            "wind_speed_kmh": tab_res["wind_speed_kmh"],
            "pressure_hpa": tab_res["pressure_hpa"],
            "source": "P3 LightGBM Tabular Model",
            "probabilities": tab_res.get("probabilities", {}),
        }
    else:
        cls_img = get_image_classifier().classify_image(img)
        classification = {
            "category": cls_img["category"],
            "confidence": cls_img["confidence"],
            "wind_speed_kmh": None,
            "pressure_hpa": None,
            "source": "P3 ResNet18 Image Model",
        }

    # 3. P4 Forecasting (if 5-step history available)
    forecast = None
    if history_obs and len(history_obs) >= 5:
        analyzer = CycloneAnalyzer()
        req = AnalyzeRequest(history=history_obs, meta=meta)
        fc_res = analyzer.forecast(req)
        forecast = fc_res.get("forecast")

    return {
        "status": "success",
        "detection": {
            "detected": det["detected"],
            "confidence": det["confidence"],
            "structural_pattern": det["structural_pattern"],
            "pattern_confidence": det["pattern_confidence"],
            "candidate_category": det["category"],
            "candidate_category_confidence": det["category_confidence"],
            "model": det.get("model_source", "P2 CycloneDetector"),
        },
        "classification": classification,
        "forecast": forecast,
        "image_source": img_name,
    }


if __name__ == "__main__":
    test_img = ROOT / "data/p2_phase7_genuine/crops/p7_mosdac_FANI_2019_0000_Very_Severe_Cyclonic_Storm.png"
    result = run_inference(test_img)
    print(json.dumps(result, indent=2))
