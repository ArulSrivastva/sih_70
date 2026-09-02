"""POST /forecast tests."""

import math

from phase6.schemas.responses import ForecastItem, ForecastSuccessResponse


def test_forecast_valid_request_ok(client, valid_request):
    r = client.post("/forecast", json=valid_request)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["model"]["experiment_id"] == "EXP005"
    assert body["input"] == {"history_hours": 24, "history_steps": 5,
                             "feature_count": 16}


def test_forecast_exactly_three_entries(client, valid_request):
    fc = client.post("/forecast", json=valid_request).json()["forecast"]
    assert len(fc) == 3
    assert [f["hours"] for f in fc] == [6, 12, 24]


def test_forecast_fields_present(client, valid_request):
    fc = client.post("/forecast", json=valid_request).json()["forecast"]
    for f in fc:
        assert set(f.keys()) == {"hours", "latitude", "longitude",
                                 "wind_speed_kmh"}


def test_forecast_physical_ranges(client, valid_request):
    fc = client.post("/forecast", json=valid_request).json()["forecast"]
    for f in fc:
        assert -90.0 <= f["latitude"] <= 90.0
        assert 0.0 <= f["longitude"] < 360.0
        assert f["wind_speed_kmh"] >= 0.0
        assert math.isfinite(f["latitude"])
        assert math.isfinite(f["longitude"])
        assert math.isfinite(f["wind_speed_kmh"])


def test_forecast_response_schema(client, valid_request):
    body = client.post("/forecast", json=valid_request).json()
    parsed = ForecastSuccessResponse(**body)
    assert parsed.status == "success"
    assert [i.hours for i in parsed.forecast] == [6, 12, 24]


def test_forecast_items_schema(client, valid_request):
    fc = client.post("/forecast", json=valid_request).json()["forecast"]
    for item in fc:
        ForecastItem(**item)


def test_forecast_reproducible_structure(client, valid_request):
    a = client.post("/forecast", json=valid_request).json()
    b = client.post("/forecast", json=valid_request).json()
    assert a == b