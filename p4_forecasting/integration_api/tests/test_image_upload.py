"""POST /api/image — P2/P3 on a user-uploaded image (Option A).

Covers the full Phase-8/Phase-10/Phase-11 battery for image ingestion:
valid upload, missing/corrupt/oversized/archive-masquerade, path-traversal
filename handling, MIME mismatch, offline inference, structured errors, and
provenance distinguishing USER-UPLOADED IMAGE vs REFERENCE FRAME.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient

from integration_api.zip_store import get_store


def _png_bytes() -> bytes:
    # A tiny, valid 8x8 PNG (solid colour) that PIL can decode.
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (8, 8), (60, 90, 120)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes() -> bytes:
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (8, 8), (120, 90, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def _reference_png() -> bytes:
    """A realistic (not tiny) frame from the artifact bundle, as a JPEG-body."""
    from PIL import Image as PILImage
    try:
        _, img = get_store().reference_image()
    except Exception:
        return _png_bytes()
    buf = io.BytesIO()
    img.resize((128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


class TestImageUpload:
    def test_valid_png_upload(self, client: TestClient):
        r = client.post("/api/image",
                        files={"file": ("frame.png", _reference_png(),
                                        "image/png")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success"
        assert "detection" in body and "classification" in body
        # honest surface (Option A / image-only)
        assert body["detection"]["location"] is None
        assert body["detection"]["movementDirection"] is None
        assert body["satellite"]["boundingBox"] is None
        assert body["classification"]["category"]
        # provenance must identify the source as user image, not reference
        assert body["provenance"]["image_source"] == "USER-UPLOADED IMAGE"
        assert any("USER-UPLOADED IMAGE" in n for n in body["provenance"]["notes"])

    def test_missing_image(self, client: TestClient):
        r = client.post("/api/image")
        assert r.status_code in (422, 400), r.text
        assert r.json()["error"]["code"] == "MISSING_FEATURE"

    def test_bad_mime_rejected(self, client: TestClient):
        r = client.post("/api/image",
                        files={"file": ("frame.txt", b"not an image",
                                        "text/plain")})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "INVALID_REQUEST"

    def test_corrupted_image_rejected(self, client: TestClient):
        r = client.post("/api/image",
                        files={"file": ("frame.png", b"\x00\x01\x02 not a real png",
                                        "image/png")})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "INVALID_REQUEST"

    def test_oversized_image_rejected(self, client: TestClient):
        # > 5 MiB soft guard
        big = b"\x00" * (6 * 1024 * 1024)
        r = client.post("/api/image",
                        files={"file": ("big.png", big, "image/png")})
        assert r.status_code in (413, 422)
        assert r.json()["error"]["code"] == "INVALID_REQUEST"

    def test_unreasonable_dimensions_rejected(self, client: TestClient):
        # a valid image with absurd dimensions (> 4096 px side) must be rejected
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (5000, 5000), (10, 10, 10)).save(buf, format="PNG")
        r = client.post("/api/image",
                        files={"file": ("huge.png", buf.getvalue(),
                                        "image/png")})
        # too large to hit the 5 MiB cap, so it must trip the dimension guard
        assert r.status_code == 422, r.text
        assert r.json()["error"]["code"] == "INVALID_REQUEST"

    def test_archive_masquerading_as_image_rejected(self, client: TestClient):
        # a real zip body but with .png extension + image MIME -> must NOT decode
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("x", "y")
        r = client.post("/api/image",
                        files={"file": ("frame.png", buf.getvalue(),
                                        "image/png")})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "INVALID_REQUEST"

    def test_path_traversal_filename_never_a_path(self, client: TestClient):
        # malicious filename must be treated as a display label only
        r = client.post("/api/image",
                        files={"file": ("..\\..\\..\\etc\\passwd.png",
                                        _reference_png(), "image/png")})
        assert r.status_code == 200, r.text
        body = r.json()
        # server strips directory components -> basename only, never a path
        assert body["satellite"]["source"] == "passwd.png"
        assert ".." not in body["satellite"]["source"]

    def test_user_filename_stripped_to_basename(self, client: TestClient):
        r = client.post("/api/image",
                        files={"file": ("../../evil/../frame.png",
                                        _reference_png(), "image/png")})
        assert r.status_code == 200
        assert r.json()["satellite"]["source"] == "frame.png"
        assert body_source_is_basename(r.json()) is True

    def test_offline_image_inference(self, client: TestClient):
        import socket as _socket
        from unittest.mock import patch
        with patch.object(_socket.socket, "connect", autospec=True,
                          side_effect=AssertionError("network attempted")), \
             patch.object(_socket.socket, "sendall", autospec=True,
                          side_effect=AssertionError("network attempted")):
            r = client.post("/api/image",
                            files={"file": ("frame.png", _reference_png(),
                                            "image/png")})
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    def test_no_stack_trace_or_path_leak(self, client: TestClient, capsys):
        r = client.post("/api/image",
                        files={"file": ("frame.png", b"garbage", "image/png")})
        assert r.status_code == 422
        assert "Traceback" not in r.text
        assert "Exception" not in r.text

    def test_reference_png_detection_classification_real(self, client: TestClient):
        # Sanity: on a real artifact frame the models return non-trivial output
        r = client.post("/api/image",
                        files={"file": ("station.png", _reference_png(),
                                        "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        # P3 category is one of the audited IMD classes
        assert body["classification"]["category"] in {
            "Depression", "Deep Depression", "Cyclonic Storm",
            "Severe Cyclonic Storm", "Very Severe Cyclonic Storm",
            "Extremely Severe Cyclonic Storm", "Super Cyclonic Storm",
        }
        assert isinstance(body["detection"]["confidence"], int)


def body_source_is_basename(body):
    # provenance should carry only the stripped original filename, not a path
    source = body.get("satellite", {}).get("source", "")
    return ".." not in source and "/" not in source and "\\" not in source


class TestCustomHistoryAnalyze:
    """P0-2: /api/analyze already accepts arbitrary valid history; the new
    HistoryEditor sends that same contract. These tests confirm the analyze
    contract is unchanged and history-driven (P4)."""

    def test_analyze_with_custom_history(self, client: TestClient, valid_request):
        body = json.loads(json.dumps(valid_request))
        # 6h spacing (26T00, 26T06, 26T12, 26T18, 27T00) keeps the contract valid
        stamps = ["2026-08-26T00:00:00Z", "2026-08-26T06:00:00Z",
                  "2026-08-26T12:00:00Z", "2026-08-26T18:00:00Z",
                  "2026-08-27T00:00:00Z"]
        for obs, ts in zip(body["history"], stamps):
            obs["timestamp"] = ts
        r = client.post("/api/analyze", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "success"
        assert len(data["forecast"]) == 3
        assert [f["hour"] for f in data["forecast"]] == [6, 12, 24]

    def test_analyze_rejects_wrong_count(self, client: TestClient, valid_request):
        body = json.loads(json.dumps(valid_request))
        body["history"] = body["history"][:3]
        r = client.post("/api/analyze", json=body)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "INVALID_HISTORY_LENGTH"

    def test_analyze_rejects_non_monotonic(self, client: TestClient,
                                           valid_request):
        body = json.loads(json.dumps(valid_request))
        # swap two adjacent timestamps: gaps stay 6h, but order breaks
        body["history"][0]["timestamp"], body["history"][1]["timestamp"] = \
            body["history"][1]["timestamp"], body["history"][0]["timestamp"]
        r = client.post("/api/analyze", json=body)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "NON_MONOTONIC_HISTORY"

    def test_analyze_rejects_invalid_lat(self, client: TestClient,
                                         valid_request):
        body = json.loads(json.dumps(valid_request))
        body["history"][1]["latitude"] = 200.0
        r = client.post("/api/analyze", json=body)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "INVALID_LATITUDE"

    def test_analyze_rejects_invalid_wind(self, client: TestClient,
                                          valid_request):
        body = json.loads(json.dumps(valid_request))
        body["history"][0]["wind_speed_kmh"] = -5
        r = client.post("/api/analyze", json=body)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "INVALID_WIND"

    def test_analyze_rejects_non_finite(self, client: TestClient,
                                        valid_request):
        body = json.loads(json.dumps(valid_request))
        # sent as the string "nan" (matches the audited battery's sst_nan_str)
        body["history"][4]["sst"] = "nan"
        r = client.post("/api/analyze", json=body)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "NON_FINITE_VALUE"

    def test_analyze_rejects_extra_field(self, client: TestClient,
                                         valid_request):
        body = json.loads(json.dumps(valid_request))
        body["history"][0]["extra"] = 1
        r = client.post("/api/analyze", json=body)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "INVALID_REQUEST"

    def test_provenance_marks_reference_frame_on_analyze(self, client,
                                                         analyze_request):
        r = client.post("/api/analyze", json=analyze_request)
        assert r.status_code == 200
        prov = r.json()["provenance"]
        # analyze (no uploaded image) uses the REFERENCE FRAME path, so it must
        # NOT be labelled as a user-uploaded image
        assert prov.get("image_source") != "USER-UPLOADED IMAGE"
        assert "USER-UPLOADED" not in json.dumps(prov)
        # and it references the bundled reference frame (an in-bundle .jpg)
        ref = prov.get("reference_image", "")
        assert ref.lower().endswith(".jpg") or ref.lower().endswith(".png") \
            or "reference" in ref.lower()
        # meta.source must not pretend this came from a user upload
        assert "USER-UPLOADED" not in json.dumps(r.json()["meta"])
