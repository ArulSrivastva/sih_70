"""GET /model tests: metadata must be read from artifacts, not hard-coded."""

import math

from phase6.schemas.responses import ModelInfoResponse


def test_model_endpoint_ok(client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["experiment_id"] == "EXP005"
    assert body["model"] == "GRU"
    assert body["loss"] == "Huber"


def test_model_contract_values(client):
    body = client.get("/model").json()
    assert body["history_steps"] == 5
    assert body["history_hours"] == 24
    assert body["feature_count"] == 16
    assert body["horizons"] == [6, 12, 24]
    assert body["input_size"] == 16
    assert body["output_size"] == 9
    assert body["targets"] == ["lat", "lon", "wind_speed_kmh"]


def test_model_primary_score_from_artifact(client):
    body = client.get("/model").json()
    # audited Phase-4 validation primary for EXP005
    assert math.isclose(body["validation_primary_score"],
                        113.07414084856835, rel_tol=1e-9)


def test_model_endpoint_reads_actual_champion(client):
    # cross-check against the on-disk champion model json
    import json
    import math
    from phase5.config import default_paths
    meta = json.loads(default_paths().champion_meta.read_text(encoding="utf-8"))
    body = client.get("/model").json()
    assert body["experiment_id"] == meta["experiment_id"]
    assert math.isclose(body["validation_primary_score"],
                        meta["primary_score"], rel_tol=1e-9)


def test_model_schema_valid(client):
    ModelInfoResponse(**client.get("/model").json())