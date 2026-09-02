"""Individual /api/detect, /api/classify, /api/forecast endpoints."""

from fastapi.testclient import TestClient


def test_detect_endpoint(client: TestClient, analyze_request):
    r = client.post("/api/detect", json=analyze_request)
    assert r.status_code == 200
    d = r.json()
    assert {"detected", "confidence", "location", "movementDirection",
            "movementSpeedKmh"} <= set(d)


def test_classify_endpoint(client: TestClient, analyze_request):
    r = client.post("/api/classify", json=analyze_request)
    assert r.status_code == 200
    c = r.json()
    assert {"category", "scale", "windSpeedKmh", "pressureHpa", "confidence",
            "structuralPattern"} <= set(c)
    assert c["scale"] == "IMD"


def test_forecast_endpoint(client: TestClient, analyze_request):
    r = client.post("/api/forecast", json=analyze_request)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["model"]["experiment_id"] == "EXP005"
    fc = body["forecast"]
    assert len(fc) == 3
    assert [f["hour"] for f in fc] == [6, 12, 24]
    for f in fc:
        assert f["label"] == f"+{f['hour']}h"
        assert isinstance(f["lat"], float)
        assert isinstance(f["lon"], float)
        assert f["windSpeedKmh"] > 0


def test_forecast_endpoint_matches_phase6(client: TestClient, analyze_request):
    """Field-mapped forecast must mirror the audited phase6 output values."""
    r = client.post("/api/forecast", json=analyze_request)
    raw = client.post("/forecast",
                      json={"history": analyze_request["history"]})
    assert raw.status_code == r.status_code == 200
    mapped = r.json()["forecast"]
    ref = raw.json()["forecast"]
    for m, x in zip(mapped, ref):
        assert m["lat"] == round(x["latitude"], 4)
        assert m["lon"] == round(x["longitude"], 4)
        assert m["windSpeedKmh"] == round(x["wind_speed_kmh"], 1)