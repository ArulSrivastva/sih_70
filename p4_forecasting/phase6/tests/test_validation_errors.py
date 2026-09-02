"""Invalid-request validation tests: every bad input must yield HTTP 422
with a structured {status, error{code, message}} body, before inference."""

import pytest


def _obs(**over):
    base = {"timestamp": "2024-08-25T00:00:00Z", "latitude": 23.30,
            "longitude": 68.50, "wind_speed_kmh": 55.6, "pressure_hpa": 994.0,
            "sst": 28.67, "wind_u": 8.5, "wind_v": 1.3}
    base.update(over)
    return base


def _request(steps):
    return {"history": steps}


def _raw_json(payload):
    import json
    return json.dumps(payload, allow_nan=True)


def _valid_steps():
    times = ["2024-08-25T00:00:00Z", "2024-08-25T06:00:00Z",
             "2024-08-25T12:00:00Z", "2024-08-25T18:00:00Z",
             "2024-08-26T00:00:00Z"]
    return [_obs(timestamp=t) for t in times]


def _expect(client, payload, code, raw=None):
    if raw is None:
        r = client.post("/forecast", json=payload)
    else:
        r = client.post("/forecast", content=raw,
                        headers={"Content-Type": "application/json"})
    assert r.status_code == 422, (r.status_code, r.text)
    body = r.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == code, body
    assert isinstance(body["error"]["message"], str) and \
        body["error"]["message"]


def test_four_steps_422(client):
    _expect(client, _request(_valid_steps()[:4]), "INVALID_HISTORY_LENGTH")


def test_six_steps_422(client):
    steps = _valid_steps()
    steps.append(_obs(timestamp="2024-08-26T06:00:00Z"))
    _expect(client, _request(steps), "INVALID_HISTORY_LENGTH")


def test_nan_422(client):
    steps = _valid_steps()
    steps[2]["wind_speed_kmh"] = float("nan")
    _expect(client, None, "NON_FINITE_VALUE",
            raw=_raw_json(_request(steps)))


def test_plus_inf_422(client):
    steps = _valid_steps()
    steps[1]["latitude"] = float("inf")
    _expect(client, None, "NON_FINITE_VALUE",
            raw=_raw_json(_request(steps)))


def test_neg_inf_422(client):
    steps = _valid_steps()
    steps[4]["sst"] = float("-inf")
    _expect(client, None, "NON_FINITE_VALUE",
            raw=_raw_json(_request(steps)))


def test_malformed_timestamp_422(client):
    steps = _valid_steps()
    steps[0]["timestamp"] = "not-a-timestamp"
    _expect(client, _request(steps), "INVALID_TIMESTAMP")


def test_wrong_spacing_422(client):
    steps = _valid_steps()
    steps[3]["timestamp"] = "2024-08-25T17:00:00Z"   # gap to 2nd entry: 11h
    _expect(client, _request(steps), "INVALID_HISTORY_SPACING")


def test_non_monotonic_422(client):
    steps = _valid_steps()[::-1]
    _expect(client, _request(steps), "NON_MONOTONIC_HISTORY")


def test_missing_feature_422(client):
    steps = _valid_steps()
    del steps[2]["wind_v"]
    _expect(client, _request(steps), "MISSING_FEATURE")


def test_missing_entire_observation_422(client):
    steps = _valid_steps()
    del steps[1]
    _expect(client, _request(steps), "INVALID_HISTORY_LENGTH")


def test_impossible_latitude_422(client):
    steps = _valid_steps()
    steps[0]["latitude"] = 95.0
    _expect(client, _request(steps), "INVALID_LATITUDE")
    steps = _valid_steps()
    steps[4]["latitude"] = -95.0
    _expect(client, _request(steps), "INVALID_LATITUDE")


def test_invalid_longitude_422(client):
    steps = _valid_steps()
    steps[1]["longitude"] = -10.0
    _expect(client, _request(steps), "INVALID_LONGITUDE")
    steps = _valid_steps()
    steps[1]["longitude"] = 360.0
    _expect(client, _request(steps), "INVALID_LONGITUDE")


def test_negative_wind_422(client):
    steps = _valid_steps()
    steps[3]["wind_speed_kmh"] = -5.0
    _expect(client, _request(steps), "INVALID_WIND")


def test_malformed_json_422(client):
    _expect(client, None, "INVALID_REQUEST",
            raw='{"history": [ }}')


def test_empty_body_422(client):
    _expect(client, None, "INVALID_REQUEST", raw="")


def test_unknown_extra_field_422(client):
    steps = _valid_steps()
    steps[0]["future_hack"] = 123.0
    _expect(client, _request(steps), "INVALID_REQUEST")


def test_no_internal_traceback_leaked(client):
    r = client.post("/forecast", content='{"history": [', 
                    headers={"Content-Type": "application/json"})
    assert r.status_code in (422, 500)
    assert "Traceback" not in r.text
    assert r.text.count(".py") == 0