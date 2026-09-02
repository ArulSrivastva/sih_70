"""Phase-6 pytest configuration.

Write-free by design: nothing outside ``p4_forecasting/phase6`` may be
created or modified.  Bytecode and pytest cache writing are disabled.
"""

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
    from phase6.api.app import create_app
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
    from phase6.schemas.requests import ForecastRequest
    return ForecastRequest.model_validate(valid_request)


@pytest.fixture(scope="session")
def adapter():
    from phase6.integration.forecasting_adapter import ForecastingAdapter
    return ForecastingAdapter()