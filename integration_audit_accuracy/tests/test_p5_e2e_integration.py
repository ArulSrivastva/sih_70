"""integration_audit_accuracy / tests / test_p5_e2e_integration.py

Comprehensive end-to-end integration test suite verifying that real P2 (E9-2),
P3 (LightGBM tabular + ResNet18 image), and P4 (CatBoost/GRU EXP005) models
execute deterministically and without mocks through FastAPI.
"""

import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
P4_DIR = ROOT / "p4_forecasting"
if str(P4_DIR) not in sys.path:
    sys.path.insert(0, str(P4_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from integration_api.app import create_app
from integration_api.p2_detector import get_detector
from integration_api.p3_classifier import get_image_classifier, get_tabular_classifier, tabular_status
from integration_api.schemas import AnalyzeRequest
from phase6.integration.forecasting_adapter import ForecastingAdapter
from phase6.schemas.requests import ForecastRequest
from src.inference import run_inference

ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_WEIGHTS = ROOT / "models/detection/model_weights.pt"
E9_2_WEIGHTS = ROOT / "models/detection/model_weights_phase9_E9_2.pt"
GENUINE_CROP = ROOT / "data/p2_phase7_genuine/crops/p7_mosdac_FANI_2019_0000_Very_Severe_Cyclonic_Storm.png"
FORECAST_REQ_PATH = ROOT / "p4_forecasting/phase6/examples/forecast_request.json"


@pytest.fixture(scope="module")
def api_client():
    app = create_app()
    return TestClient(app)


def test_baseline_weights_immutability():
    """Verify that models/detection/model_weights.pt is byte-identical to locked baseline."""
    assert BASELINE_WEIGHTS.exists(), "Baseline weights file must exist"
    h = hashlib.sha256(BASELINE_WEIGHTS.read_bytes()).hexdigest()
    assert h == "c296aa21f3e105847878a67abe69390b4a0c566ff31011c08abf78154c2e1971", \
        f"BASELINE MODEL TAMPERED! SHA-256 is {h}"


def test_p2_e9_2_candidate_loading_and_inference():
    """Verify that E9-2 candidate is loaded and produces expected predictions on genuine crop."""
    assert E9_2_WEIGHTS.exists(), "E9-2 candidate checkpoint must exist"
    det = get_detector()
    status = det.status()
    assert status["is_candidate"] is True, "Detector should prioritize E9-2 candidate"
    assert "model_weights_phase9_E9_2.pt" in status["model"]

    img = Image.open(GENUINE_CROP).convert("RGB")
    res = det.detect(img)
    assert res["detected"] is True
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["structural_pattern"] in ["curved_band", "eye_visible", "shear_pattern"]
    assert res["category"] in ["Cyclonic Storm", "Deep Depression", "Depression", "Very Severe Cyclonic Storm"]


def test_p3_tabular_lightgbm_inference():
    """Verify that P3 LightGBM tabular model unpickles and predicts intensity."""
    tab_stat = tabular_status()
    assert tab_stat["available"] is True, f"Tabular model not available: {tab_stat}"

    clf = get_tabular_classifier()
    # Test on severe cyclonic conditions
    res = clf.classify_tabular(lat=16.52, lon=82.31, sst=28.6, pressure_msl=950.0, wind_u=11.8, wind_v=6.7)
    assert "category" in res
    assert "confidence" in res
    assert "wind_speed_kmh" in res
    assert "pressure_hpa" in res
    assert res["confidence"] > 0.5
    assert res["wind_speed_kmh"] > 100.0


def test_p4_forecasting_adapter():
    """Verify that P4 forecasting adapter produces +6h, +12h, +24h displacements."""
    with open(FORECAST_REQ_PATH, "r", encoding="utf-8") as f:
        req_data = json.load(f)
    req = ForecastRequest(**req_data)
    fa = ForecastingAdapter()
    res = fa.forecast(req)
    assert res["status"] == "success"
    assert len(res["forecast"]) == 3
    hours = [f["hours"] for f in res["forecast"]]
    assert hours == [6, 12, 24]


def test_api_health_endpoint(api_client):
    """Verify /api/health reports all ML components ready."""
    resp = api_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_ready"] is True
    assert data["ml"]["p2"]["ready"] is True
    assert data["ml"]["p2"]["is_candidate"] is True
    assert data["ml"]["p3_tabular"]["available"] is True


def test_api_analyze_e2e_contract(api_client):
    """Verify /api/analyze accepts history and returns complete dashboard contract."""
    with open(FORECAST_REQ_PATH, "r", encoding="utf-8") as f:
        req_data = json.load(f)
    resp = api_client.post("/api/analyze", json={"history": req_data["history"]})
    assert resp.status_code == 200
    data = resp.json()

    # Verify essential contract keys
    assert "meta" in data
    assert "detection" in data
    assert "classification" in data
    assert "forecast" in data
    assert "landfall" in data
    assert "risk" in data
    assert "provenance" in data

    assert data["detection"]["detected"] in [True, False]
    assert data["classification"]["scale"] == "IMD"
    assert len(data["forecast"]) == 3


def test_api_image_upload_e2e(api_client):
    """Verify /api/image runs real P2 candidate inference on uploaded satellite crop."""
    with open(GENUINE_CROP, "rb") as f:
        img_bytes = f.read()
    files = {"file": ("fani.png", io.BytesIO(img_bytes), "image/png")}
    resp = api_client.post("/api/image", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["detection"]["detected"] is True
    assert data["detection"]["confidence"] >= 50
    assert data["classification"]["scale"] == "IMD"
    assert "p7_mosdac_FANI" in data["provenance"]["sources"]["detection"] or "model_weights_phase9_E9_2.pt" in data["provenance"]["sources"]["detection"]


def test_standalone_cli_inference():
    """Verify src/inference.py run_inference function works end-to-end."""
    res = run_inference(GENUINE_CROP, history_input=FORECAST_REQ_PATH)
    assert res["status"] == "success"
    assert res["detection"]["detected"] is True
    assert res["classification"]["category"] is not None
    assert res["forecast"] is not None
    assert len(res["forecast"]) == 3
