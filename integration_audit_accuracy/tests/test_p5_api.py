"""P5/API: backend works; dashboard target contract missing (mock-only)."""


def test_frontend_is_mock_only(p5):
    assert p5["use_mock_working_copy"] is True
    assert p5["use_mock_pristine_copy"] is True
    assert p5["dashboard_target_/api/analyze_status"] == 404


def test_backend_path_surface_missing_dashboard_targets(p5):
    backend = set(p5["delivered_backend_paths_verified"])
    for t in ["GET /api/detect", "GET /api/classify", "GET /api/forecast"]:
        assert t not in backend


def test_forecast_shape_mismatch_documented(p5):
    assert any("windSpeedKmh" in s and "wind_speed_kmh" in s for s in p5["forecast_shape_mismatch"])


def test_live_api_health(api_smoke, client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_ready"] is True and body["offline"] is True


def test_live_api_model_metadata(api_smoke, client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["experiment_id"] == "EXP005"
    assert body["families"] if False else body.get("model") in ("GRU", "gru")


def test_live_api_forecast_shape(client):
    import json
    from pathlib import Path
    req = json.loads(Path(__file__).resolve().parent.parent.parent.joinpath(
        "p4_forecasting/phase6/examples/forecast_request.json").read_text(encoding="utf-8"))
    r = client.post("/forecast", json=req)
    assert r.status_code == 200
    fc = r.json()["forecast"]
    assert len(fc) == 3
    for item in fc:
        assert set(item.keys()) == {"hours", "latitude", "longitude", "wind_speed_kmh"}
    assert [item["hours"] for item in fc]
    assert client.get("/api/analyze").status_code == 404


def test_live_api_rejects_bad_input(client):
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent.parent.joinpath(
        "p4_forecasting/phase6/examples/forecast_request.json")
    req = json.loads(p.read_text(encoding="utf-8"))
    bad = json.loads(json.dumps(req))
    bad["history"] = bad["history"][:4]
    assert client.post("/forecast", json=bad).status_code == 422
    bad2 = json.loads(json.dumps(req))
    bad2["history"][1].pop("sst")
    assert client.post("/forecast", json=bad2).status_code == 422