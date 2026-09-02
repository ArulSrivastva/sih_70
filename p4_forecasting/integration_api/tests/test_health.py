"""GET /api/health — module readiness, no network, no local weights."""

from fastapi.testclient import TestClient

from integration_api.config import P4_FORECASTING, P2_WEIGHTS_REL, P3_IMAGE_WEIGHTS_REL


def test_health_ok(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "cyclone-integration"
    assert body["phase"] == "integration_api"
    assert body["offline"] is True
    assert body["model_ready"] is True
    assert body["ml"]["p2"]["ready"] is True
    assert body["ml"]["p3_image"]["ready"] is True
    assert body["ml"]["p3_tabular"]["available"] is False


def test_health_reports_reference_frame(client: TestClient):
    body = client.get("/api/health").json()
    assert body["ml"]["reference_frame"].endswith(".jpg")


def test_no_local_weights_extracted(client: TestClient, analyze_request):
    """Artifacts are served from the zip in memory; nothing lands on disk."""
    r = client.post("/api/analyze", json=analyze_request)
    assert r.status_code == 200
    for rel in (P2_WEIGHTS_REL, P3_IMAGE_WEIGHTS_REL):
        on_disk = (P4_FORECASTING / rel).resolve()
        assert not on_disk.exists(), f"{rel} must stay inside the zip only"