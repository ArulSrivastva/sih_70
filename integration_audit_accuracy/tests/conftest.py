"""Shared fixtures for the pre-integration audit test suite.

Runs entirely inside the audit dir; never writes outside it.
"""

import json
import sys
from pathlib import Path

import pytest

AUDIT = Path(__file__).resolve().parent.parent
P4 = AUDIT.parent / "p4_forecasting"
WORKSPACE = AUDIT.parent
ZIP = WORKSPACE / "PS70-main.zip"


@pytest.fixture(scope="session")
def p1():
    return json.loads((AUDIT / "p1_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def p2():
    return json.loads((AUDIT / "p2_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def p3():
    return json.loads((AUDIT / "p3_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def p4():
    return json.loads((AUDIT / "p4_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def p5():
    return json.loads((AUDIT / "p5_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def api_smoke():
    p = AUDIT / "tmp" / "api_smoke.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


@pytest.fixture(scope="session")
def hashes_before():
    p = AUDIT / "SOURCE_HASHES_BEFORE.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@pytest.fixture(scope="session")
def hashes_after():
    p = AUDIT / "SOURCE_HASHES_AFTER.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@pytest.fixture(scope="session")
def client():
    sys.path.insert(0, str(P4))
    from fastapi.testclient import TestClient
    from phase6.api.app import create_app
    with TestClient(create_app()) as c:
        yield c