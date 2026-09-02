"""POST /api/analyze reuses phase-6 validation — bad input is rejected 422."""

import json

from fastapi.testclient import TestClient


def test_analyze_rejects_4_steps(client: TestClient, valid_request):
    body = dict(valid_request)
    body["history"] = body["history"][:4]
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_HISTORY_LENGTH"


def test_analyze_rejects_bad_spacing(client: TestClient, valid_request):
    body = json.loads(json.dumps(valid_request))
    body["history"][3]["timestamp"] = "2024-08-25T17:00:00Z"
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_HISTORY_SPACING"


def test_analyze_rejects_unknown_extra(client: TestClient, valid_request):
    body = json.loads(json.dumps(valid_request))
    body["surprise"] = 1
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_REQUEST"


def test_analyze_rejects_bad_meta_extra(client: TestClient, analyze_request):
    body = json.loads(json.dumps(analyze_request))
    body["meta"]["nope"] = True
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 422


def test_analyze_rejects_malformed_json(client: TestClient):
    r = client.post("/api/analyze", content='{"history": [ }}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_REQUEST"


def test_phase6_endpoints_still_work(client: TestClient, valid_request):
    """Composition must not break the audited phase-6 routes."""
    assert client.get("/health").status_code == 200
    r = client.post("/forecast", json=valid_request)
    assert r.status_code == 200
    assert len(r.json()["forecast"]) == 3