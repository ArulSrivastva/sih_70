"""Write-free by design: nothing outside p4_forecasting/integration_api may be
created or modified.  Bytecode and pytest cache writing are disabled."""

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import pathlib

_PKG = pathlib.Path(__file__).resolve().parent.parent.parent  # p4_forecasting
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import json

import pytest


@pytest.fixture(scope="session")
def app():
    from integration_api.app import create_app
    return create_app()


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def valid_request():
    ex = pathlib.Path(_PKG) / "phase6" / "examples" / "forecast_request.json"
    return json.loads(ex.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def valid_request_model(valid_request):
    from integration_api.schemas import AnalyzeRequest
    return AnalyzeRequest.model_validate(valid_request)


@pytest.fixture(scope="session")
def analyze_request(valid_request):
    body = dict(valid_request)
    body["meta"] = {
        "systemId": "BOB07",
        "systemName": "Cyclonic Storm ANIKA",
        "basin": "Bay of Bengal",
        "lastPass": "2026-08-26T05:30:00Z",
    }
    return body


@pytest.fixture(scope="session")
def adapter():
    from phase6.integration.forecasting_adapter import ForecastingAdapter
    return ForecastingAdapter()