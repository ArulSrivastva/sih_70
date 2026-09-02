"""GET /health tests."""

from phase6.schemas.responses import HealthResponse


def test_health_endpoint_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "cyclone-forecasting"
    assert body["phase"] == "phase6"
    assert body["offline"] is True


def test_health_model_ready_true_with_artifacts(client):
    body = client.get("/health").json()
    assert body["model_ready"] is True
    assert "model_ready" in body


def test_health_schema_valid(client):
    body = client.get("/health").json()
    parsed = HealthResponse(**body)
    assert parsed.status == "ok"
    assert parsed.offline is True


def test_health_no_network(client):
    # health must never issue network calls; allow any status shape but the
    # payload must say offline
    body = client.get("/health").json()
    assert body["offline"] is True