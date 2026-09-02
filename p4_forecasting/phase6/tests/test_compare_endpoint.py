"""POST /forecast/compare tests."""

from phase6.schemas.responses import CompareSuccessResponse


def test_compare_returns_all_three_forecasts(client, valid_request):
    r = client.post("/forecast/compare", json=valid_request)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    for key in ("model_forecast", "persistence_forecast",
                "movement_vector_forecast"):
        assert key in body


def test_compare_correct_horizons(client, valid_request):
    body = client.post("/forecast/compare", json=valid_request).json()
    for key in ("model_forecast", "persistence_forecast",
                "movement_vector_forecast"):
        fc = body[key]
        assert len(fc) == 3
        assert [f["hours"] for f in fc] == [6, 12, 24]
        for f in fc:
            assert set(f.keys()) == {"hours", "latitude", "longitude",
                                     "wind_speed_kmh"}


def test_compare_model_matches_forecast_route(client, valid_request):
    model_fc = client.post("/forecast", json=valid_request).json()["forecast"]
    cmp = client.post("/forecast/compare", json=valid_request).json()
    assert cmp["model_forecast"] == model_fc


def test_compare_persistence_is_latest_observation(client, valid_request):
    import pytest
    cmp = client.post("/forecast/compare", json=valid_request).json()
    last = valid_request["history"][-1]
    for f in cmp["persistence_forecast"]:
        assert f["latitude"] == pytest.approx(last["latitude"], abs=1e-4)
        assert f["longitude"] == pytest.approx(last["longitude"], abs=1e-4)
        assert f["wind_speed_kmh"] == pytest.approx(last["wind_speed_kmh"],
                                                    abs=1e-4)


def test_compare_response_schema(client, valid_request):
    body = client.post("/forecast/compare", json=valid_request).json()
    parsed = CompareSuccessResponse(**body)
    assert parsed.status == "success"
    assert len(parsed.persistence_forecast) == 3


def test_compare_model_metadata(client, valid_request):
    body = client.post("/forecast/compare", json=valid_request).json()
    assert body["model"]["experiment_id"] == "EXP005"


def test_compare_physical_ranges(client, valid_request):
    body = client.post("/forecast/compare", json=valid_request).json()
    for key in ("model_forecast", "persistence_forecast",
                "movement_vector_forecast"):
        for f in body[key]:
            assert -90.0 <= f["latitude"] <= 90.0
            assert 0.0 <= f["longitude"] < 360.0
            assert f["wind_speed_kmh"] >= 0.0