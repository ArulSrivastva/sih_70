"""POST /api/analyze — full dashboard contract, field by field."""

from fastapi.testclient import TestClient

REQUIRED_TOP_LEVEL = [
    "meta", "detection", "classification", "forecast", "landfall", "risk",
    "historicalTrack", "windHistory", "pressureHistory", "confidenceHistory",
    "sstHistory", "envWindHistory", "satellite", "provenance",
]


def test_analyze_returns_200_and_contract(client: TestClient, analyze_request):
    r = client.post("/api/analyze", json=analyze_request)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    for key in REQUIRED_TOP_LEVEL:
        assert key in body, f"missing top-level key {key}"


def test_analyze_meta(client: TestClient, analyze_request):
    body = client.post("/api/analyze", json=analyze_request).json()
    meta = body["meta"]
    assert meta["systemId"] == "BOB07"
    assert meta["systemName"] == "Cyclonic Storm ANIKA"
    assert meta["basin"] == "Bay of Bengal"
    assert meta["lastPass"] == "2026-08-26T05:30:00Z"
    assert "PS70-main.zip" in meta["source"] or "frame" in meta["source"]


def test_analyze_defaults_meta_when_absent(client: TestClient, valid_request):
    body = client.post("/api/analyze", json=dict(valid_request)).json()
    assert body["meta"]["systemId"] == "BOB07"


def test_analyze_detection(client: TestClient, analyze_request):
    body = client.post("/api/analyze", json=analyze_request).json()
    d = body["detection"]
    assert isinstance(d["detected"], bool)
    assert 0 <= d["confidence"] <= 100
    assert "lat" in d["location"] and "lon" in d["location"]
    assert isinstance(d["movementDirection"], str)
    assert d["movementSpeedKmh"] >= 0

    last = analyze_request["history"][-1]
    assert d["location"]["lat"] == round(last["latitude"], 4)
    assert d["location"]["lon"] == round(last["longitude"], 4)


def test_analyze_classification(client: TestClient, analyze_request):
    body = client.post("/api/analyze", json=analyze_request).json()
    c = body["classification"]
    assert c["scale"] == "IMD"
    assert isinstance(c["category"], str) and c["category"]
    assert 0 <= c["confidence"] <= 100
    assert c["windSpeedKmh"] > 0
    assert isinstance(c["pressureHpa"], float)
    assert isinstance(c["structuralPattern"], str)

    last = analyze_request["history"][-1]
    assert c["windSpeedKmh"] == round(last["wind_speed_kmh"], 1)
    assert c["pressureHpa"] == round(last["pressure_hpa"], 1)


def test_analyze_forecast(client: TestClient, analyze_request):
    body = client.post("/api/analyze", json=analyze_request).json()
    fc = body["forecast"]
    assert len(fc) == 3
    assert [f["hour"] for f in fc] == [6, 12, 24]
    for f in fc:
        assert f["label"] == f"+{f['hour']}h"
        assert isinstance(f["lat"], float)
        assert isinstance(f["lon"], float)
        assert f["windSpeedKmh"] > 0
        assert f["pressureHpa"] is None   # honest: no calibrated pressure
        assert f["confidence"] is None    # honest: no calibrated uncertainty


def test_analyze_landfall_and_risk(client: TestClient, analyze_request):
    body = client.post("/api/analyze", json=analyze_request).json()
    lf = body["landfall"]
    assert "estimated" in lf and lf["estimated"] in (True, False)
    if lf["estimated"]:
        assert lf["latitude"] is not None and lf["longitude"] is not None
        assert lf["estimated_time"]
        assert lf["predictedWindKmh"] is not None
    risk = body["risk"]
    assert 0 <= risk["score"] <= 100
    assert risk["level"] in ("HIGH", "MODERATE", "LOW")


def test_analyze_histories(client: TestClient, analyze_request):
    body = client.post("/api/analyze", json=analyze_request).json()
    for key in ("historicalTrack", "windHistory", "pressureHistory",
                "confidenceHistory", "sstHistory", "envWindHistory"):
        assert len(body[key]) == 5, key
    hist = body["historicalTrack"]
    inp = analyze_request["history"]
    assert [p["lat"] for p in hist] == [round(o["latitude"], 4)
                                        for o in inp]
    assert [p["lon"] for p in hist] == [round(o["longitude"], 4)
                                        for o in inp]
    assert [v["t"] for v in body["windHistory"]] == [
        "-24h", "-18h", "-12h", "-6h", "Now"]
    assert [v["value"] for v in body["windHistory"]] == [
        round(o["wind_speed_kmh"], 1) for o in inp]
    assert [v["value"] for v in body["pressureHistory"]] == [
        round(o["pressure_hpa"], 1) for o in inp]
    assert [v["value"] for v in body["sstHistory"]] == [
        round(o["sst"], 2) for o in inp]


def test_analyze_satellite(client: TestClient, analyze_request):
    body = client.post("/api/analyze", json=analyze_request).json()
    sat = body["satellite"]
    assert sat["label"]
    assert sat["boundingBox"] is None   # honest: no audited localizer
    assert sat["source"].endswith(".jpg")


def test_analyze_provenance(client: TestClient, analyze_request):
    body = client.post("/api/analyze", json=analyze_request).json()
    prov = body["provenance"]
    assert "pipeline" in prov
    assert "model_weights.pt" in prov["sources"]["detection"]
    assert "image_only_model.pt" in prov["sources"]["classification"]
    assert "EXP005" in prov["sources"]["forecast"]
    assert prov["reference_image"].endswith(".jpg")
    assert prov["tabular"]["available"] is False
    assert any("NOT_RUN" in n or "not loaded" in n or "lightgbm" in n
               for n in prov["notes"])
    assert prov["notes"]  # honesty notes always present


def test_analyze_deterministic(client: TestClient, analyze_request):
    a = client.post("/api/analyze", json=analyze_request)
    b = client.post("/api/analyze", json=analyze_request)
    assert a.status_code == b.status_code == 200
    assert a.content == b.content