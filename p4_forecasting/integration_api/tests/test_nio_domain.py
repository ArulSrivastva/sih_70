"""POST /api/analyze — upstream North Indian Ocean (NIO) domain policy guard.

The phase-6 layer only enforces global physical bounds (lat in [-90, 90], lon
in [0, 360)).  The integration layer additionally rejects any observation that
falls outside the NIO box the IMD/RSMC New Delhi mandate covers:

    lat in [0, 30] deg N,  lon in [40, 100] deg E (canonical 0..360).

Out-of-NIO input is rejected 422 with code OUT_OF_DOMAIN and is NEVER silently
clamped.  Notably, a latitude of -40 (historically accepted by phase-6 as
valid, returning 200) is now rejected upstream.
"""

import json

from fastapi.testclient import TestClient


def _set_lat(valid_request, value):
    body = json.loads(json.dumps(valid_request))
    body["history"][0]["latitude"] = value
    return body


def _set_lon(valid_request, value):
    body = json.loads(json.dumps(valid_request))
    body["history"][0]["longitude"] = value
    return body


def _expect_out_of_domain(client, payload):
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 422, (r.status_code, r.text)
    assert r.json()["error"]["code"] == "OUT_OF_DOMAIN"


def _expect_ok(client, payload):
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200, (r.status_code, r.text)


# ---------------------------------------------------------------------------
# Latitude boundaries
# ---------------------------------------------------------------------------
def test_nio_lat_min_boundary_accepted(client, valid_request):
    _expect_ok(client, _set_lat(valid_request, 0.0))


def test_nio_lat_max_boundary_accepted(client, valid_request):
    _expect_ok(client, _set_lat(valid_request, 30.0))


def test_nio_lat_below_min_rejected(client, valid_request):
    _expect_out_of_domain(client, _set_lat(valid_request, -0.001))


def test_nio_lat_above_max_rejected(client, valid_request):
    _expect_out_of_domain(client, _set_lat(valid_request, 30.001))


def test_nio_lat_minus40_rejected(client, valid_request):
    """lat -40 (historically 200 through phase-6) is now OUT_OF_DOMAIN."""
    _expect_out_of_domain(client, _set_lat(valid_request, -40.0))


# ---------------------------------------------------------------------------
# Longitude boundaries
# ---------------------------------------------------------------------------
def test_nio_lon_min_boundary_accepted(client, valid_request):
    _expect_ok(client, _set_lon(valid_request, 40.0))


def test_nio_lon_max_boundary_accepted(client, valid_request):
    _expect_ok(client, _set_lon(valid_request, 100.0))


def test_nio_lon_below_min_rejected(client, valid_request):
    _expect_out_of_domain(client, _set_lon(valid_request, 39.999))


def test_nio_lon_above_max_rejected(client, valid_request):
    _expect_out_of_domain(client, _set_lon(valid_request, 100.001))


def test_nio_lon_120_passes_phase6_but_rejected_upstream(client, valid_request):
    """lon 120 is within phase-6 [0,360) yet outside the NIO box."""
    _expect_out_of_domain(client, _set_lon(valid_request, 120.0))


# ---------------------------------------------------------------------------
# Non-finite / out-of-range beats the NIO guard (field-level first)
# ---------------------------------------------------------------------------
def test_nio_nan_rejected_as_non_finite(client, valid_request):
    raw = json.dumps(valid_request).replace('"latitude": 23.3',
                                            '"latitude": NaN', 1)
    r = client.post("/api/analyze", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "NON_FINITE_VALUE"


# ---------------------------------------------------------------------------
# Multi-observation guard (any off-box point trips it)
# ---------------------------------------------------------------------------
def test_nio_guard_checks_every_observation(client, valid_request):
    body = json.loads(json.dumps(valid_request))
    body["history"][4]["latitude"] = -33.0  # last observation, off-box
    _expect_out_of_domain(client, body)


# ---------------------------------------------------------------------------
# The error must be explicit about NOT clamping
# ---------------------------------------------------------------------------
def test_nio_error_message_rejects_not_clamps(client, valid_request):
    r = client.post("/api/analyze",
                    json=_set_lat(valid_request, -40.0))
    assert r.status_code == 422
    msg = r.json()["error"]["message"]
    assert "rejected" in msg and "not clamped" in msg
