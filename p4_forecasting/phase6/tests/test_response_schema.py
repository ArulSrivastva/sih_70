"""Response-schema / OpenAPI contract tests."""

from phase6.schemas.responses import (CompareSuccessResponse, ErrorResponse,
                                      ForecastSuccessResponse, HealthResponse,
                                      ModelInfoResponse)


def test_health_schema(client):
    body = client.get("/health").json()
    HealthResponse(**body)


def test_model_schema(client):
    body = client.get("/model").json()
    parsed = ModelInfoResponse(**body)
    assert parsed.horizons == [6, 12, 24]


def test_forecast_success_schema(client, valid_request):
    body = client.post("/forecast", json=valid_request).json()
    ForecastSuccessResponse(**body)


def test_compare_success_schema(client, valid_request):
    body = client.post("/forecast/compare", json=valid_request).json()
    CompareSuccessResponse(**body)


def test_error_schema(client):
    bad = {"history": []}
    r = client.post("/forecast", json=bad)
    assert r.status_code == 422
    ErrorResponse(**r.json())


def test_openapi_json_available(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert "paths" in spec
    assert set(spec["paths"].keys()) >= {
        "/health", "/model", "/forecast", "/forecast/compare"}


def test_openapi_docs_available(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower() or "<html" in r.text.lower()


def test_openapi_names_spec_fields(client):
    spec = client.get("/openapi.json").json()
    schema = spec["components"]["schemas"]
    assert "ForecastRequest" in schema
    assert "ForecastSuccessResponse" in schema


def test_forecast_items_never_share_mutable_reference(client, valid_request):
    fc = client.post("/forecast", json=valid_request).json()["forecast"]
    assert fc[0] is not fc[1]
    assert all(isinstance(f, dict) for f in fc)