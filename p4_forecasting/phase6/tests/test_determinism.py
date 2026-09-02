"""Determinism: the exact same request must produce the exact same response
(two calls, byte-stable JSON, CPU inference)."""


def test_same_request_twice_identical(client, valid_request):
    a = client.post("/forecast", json=valid_request)
    b = client.post("/forecast", json=valid_request)
    assert a.status_code == b.status_code == 200
    assert a.json() == b.json()


def test_same_request_twice_identical_bytes(client, valid_request):
    a = client.post("/forecast", json=valid_request).content
    b = client.post("/forecast", json=valid_request).content
    assert a == b


def test_compare_deterministic(client, valid_request):
    a = client.post("/forecast/compare", json=valid_request).json()
    b = client.post("/forecast/compare", json=valid_request).json()
    assert a == b


def test_adapter_forecasts_bitwise_stable(adapter, valid_request_model):
    import math
    a = adapter.forecast(valid_request_model)
    b = adapter.forecast(valid_request_model)
    assert a == b
    for f1, f2 in zip(a["forecast"], b["forecast"]):
        assert math.isclose(f1["latitude"], f2["latitude"], rel_tol=0.0,
                            abs_tol=0.0)
    assert a == b


def test_cpu_inference_used(adapter):
    import torch
    adapter.ensure_loaded()
    assert str(next(adapter.service.predictor.model.parameters()).device) \
        == "cpu"