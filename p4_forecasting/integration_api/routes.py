"""integration_api / routes.py

Mounts the dashboard contract under ``/api``:
  GET  /api/health
  POST /api/analyze
  POST /api/detect
  POST /api/classify
  POST /api/forecast
  POST /api/image      -> P2 + P3 on a user-uploaded image (Option A)

Errors follow the phase-6 vocabulary ({status, error:{code,message}}) via
the shared AdapterError handler already registered on the base app.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import (APIRouter, File, HTTPException, Request, UploadFile)
from fastapi.responses import JSONResponse
from PIL import Image

from .analyzer import CycloneAnalyzer
from .schemas import (AnalyzeRequest, AnalyzeResponse, ClassificationBlock,
                      DetectionBlock, ForecastResponse, HealthBlock,
                      ImageAnalyzeResponse)

logger = logging.getLogger("integration_api.image")

router = APIRouter(prefix="/api", tags=["integration"])

# Allowed image uploads. JPEG/PNG only; anything else (including archive
# masquerading as an image) is rejected before any model runs.
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_ALLOWED_MIME = {"image/jpeg", "image/png"}
# Soft guard in addition to the phase-6 body size limit. The phase-6
# middleware already rejects bodies > MAX_REQUEST_BYTES (256 KB) with 413.
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
# Reasonable dimension cap: a decodable image beyond this is likely a
# decompression-bomb / distorted upload and is rejected before any model runs.
_MAX_IMAGE_DIM = 4096  # pixels per side
_MAX_IMAGE_MP = 4096 * 4096  # total pixels (16.8 MP)


def _analyzer(request: Request) -> CycloneAnalyzer:
    return request.app.state.analyzer


def _image_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"status": "error", "error": {"code": code, "message": message}},
        status_code=status)


@router.post("/image", response_model=ImageAnalyzeResponse,
             summary="P2 detection + P3 classification on a "
                     "user-uploaded satellite image")
async def analyze_image(request: Request,
                        file: Optional[UploadFile] = File(None,
                                                          description="JPEG/PNG image")) \
        -> dict:
    """Secure, offline image ingestion.

    The user-supplied filename is NEVER treated as a filesystem path: it is
    used only as a display label in provenance, and only after stripping any
    directory components.  The bytes are decoded in memory; nothing is written
    to disk and no archive is ever extracted.
    """
    if file is None:
        return _image_error(422, "MISSING_FEATURE", "no image file was provided")

    # 1) original filename -> display label only (never a path).
    filename = Path(file.filename or "").name

    # 2) extension + MIME allow-list.
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return _image_error(422, "INVALID_REQUEST",
                            "unsupported image type; only .jpg/.jpeg/.png")
    if (file.content_type or "").lower() not in _ALLOWED_MIME:
        return _image_error(422, "INVALID_REQUEST",
                            "unsupported MIME type; only image/jpeg or image/png")

    # 3) size + actual decoding (gorilla-proof: bytes must be a real image).
    raw = await file.read()
    if not raw:
        return _image_error(422, "INVALID_REQUEST", "image file is empty")
    if len(raw) > _MAX_IMAGE_BYTES:
        return _image_error(413, "INVALID_REQUEST",
                            "image exceeds the 5 MiB size limit")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image.load()
    except Exception as exc:
        logger.info("image decode failed for %r: %s", filename,
                    exc.__class__.__name__)
        return _image_error(422, "INVALID_REQUEST",
                            "file is not a decodable image")

    # 4) dimension guard (anti-decompression-bomb; reasonable size only).
    w, h = image.size
    if w <= 0 or h <= 0 or w > _MAX_IMAGE_DIM or h > _MAX_IMAGE_DIM \
            or w * h > _MAX_IMAGE_MP:
        return _image_error(422, "INVALID_REQUEST",
                            "image dimensions are out of the supported range "
                            f"(max {_MAX_IMAGE_DIM}x{_MAX_IMAGE_DIM} px)")

    try:
        payload = _analyzer(request).analyze_image(image, filename)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("image inference failed")
        return _image_error(500, "INFERENCE_ERROR",
                            "image analysis failed internally; see server log")

    return ImageAnalyzeResponse(status="success", **payload)


@router.get("/health", response_model=HealthBlock,
            summary="Integration service health + ML module readiness")
def health(request: Request) -> dict:
    return HealthBlock(**_analyzer(request).health())


@router.post("/analyze", response_model=AnalyzeResponse,
             summary="Full dashboard payload "
                     "(detection + classification + forecast + heuristics)")
def analyze(request: Request, body: AnalyzeRequest) -> dict:
    payload = _analyzer(request).analyze(body)
    return AnalyzeResponse(status="success", **payload)


@router.post("/detect", response_model=DetectionBlock,
             summary="Detection block only (P2 on reference INSAT frame)")
def detect(request: Request, body: AnalyzeRequest) -> dict:
    return DetectionBlock(**_analyzer(request).detect(body))


@router.post("/classify", response_model=ClassificationBlock,
             summary="Classification block only (P3 category/confidence + "
                     "latest observed wind/pressure)")
def classify(request: Request, body: AnalyzeRequest) -> dict:
    return ClassificationBlock(**_analyzer(request).classify(body))


@router.post("/forecast", response_model=ForecastResponse,
             summary="Forecast block only (EXP005 via phase6 adapter, field-"
                     "mapped to the dashboard shape)")
def forecast(request: Request, body: AnalyzeRequest) -> dict:
    return ForecastResponse(**_analyzer(request).forecast(body))


SAMPLE_CROPS = {
    "FANI": {
        "name": "Super Cyclone FANI (2019)",
        "file": "p7_mosdac_FANI_2019_0000_Very_Severe_Cyclonic_Storm.png",
        "category": "Extremely Severe Cyclonic Storm",
        "basin": "Bay of Bengal",
        "source": "MOSDAC INSAT-3D IR",
    },
    "AMPHAN": {
        "name": "Super Cyclone AMPHAN (2020)",
        "file": "p7_mosdac_AMPHAN_2020_0001_Very_Severe_Cyclonic_Storm.png",
        "category": "Super Cyclonic Storm",
        "basin": "Bay of Bengal",
        "source": "MOSDAC INSAT-3D IR",
    },
    "BIPARJOY": {
        "name": "Cyclone BIPARJOY (2023)",
        "file": "p7_mosdac_BIPARJOY_2023_0015_Very_Severe_Cyclonic_Storm.png",
        "category": "Very Severe Cyclonic Storm",
        "basin": "Arabian Sea",
        "source": "MOSDAC INSAT-3D IR",
    },
    "TAUKTAE": {
        "name": "Cyclone TAUKTAE (2021)",
        "file": "p7_mosdac_TAUKTAE_2021_0011_Very_Severe_Cyclonic_Storm.png",
        "category": "Extremely Severe Cyclonic Storm",
        "basin": "Arabian Sea",
        "source": "MOSDAC INSAT-3D IR",
    },
}


@router.get("/sample_images", summary="List genuine MOSDAC sample cyclone crops")
def get_sample_images() -> list:
    return [{"key": k, **v} for k, v in SAMPLE_CROPS.items()]


@router.get("/sample_images/{key}", summary="Serve genuine MOSDAC sample cyclone crop file")
def get_sample_image_file(key: str):
    from fastapi.responses import FileResponse
    key_upper = key.upper()
    if key_upper not in SAMPLE_CROPS:
        raise HTTPException(status_code=404, detail="Sample image not found")
    filename = SAMPLE_CROPS[key_upper]["file"]
    path = Path(__file__).resolve().parent.parent.parent / "data" / "p2_phase7_genuine" / "crops" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample crop file missing on disk")
    return FileResponse(str(path), media_type="image/png", filename=filename)