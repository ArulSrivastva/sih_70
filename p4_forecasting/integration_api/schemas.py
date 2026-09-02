"""integration_api / schemas.py

Wire schemas for the ``/api/*`` endpoints.

The request schema *is* the audited Phase-6 ``ForecastRequest`` (exactly 5
6-hourly observations with physical-range validation) plus an optional
``meta`` block.  The response schemas mirror the dashboard contract
(``cyclone-dashboard/src/api/mockData.js``) field for field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from phase6.schemas.requests import ForecastRequest, ValidationCode

from .config import NIO_LAT_RANGE, NIO_LON_RANGE, OUT_OF_DOMAIN


class AnalyzeMeta(BaseModel):
    """Optional identity block echoed by the API (demo defaults otherwise)."""

    model_config = ConfigDict(extra="forbid")

    systemId: str = Field(default="BOB07", description="system identifier")
    systemName: str = Field(default="Cyclonic Storm ANIKA")
    basin: str = Field(default="Bay of Bengal")
    lastPass: Optional[str] = Field(
        default=None, description="ISO-8601 timestamp of the latest pass")
    source: Optional[str] = Field(
        default=None, description="free-form frame attribution")


class AnalyzeRequest(ForecastRequest):
    """Dashboard analyze body: the audited forecast history + optional meta.

    Inherits ALL Phase-6 validation (5 observations, 6-hour spacing,
    strict physical bounds, finite values, monotonic timestamps) AND adds an
    upstream North-Indian-Ocean policy guard (lat 0..30 N, lon 40..100 E) on
    every observation.  Input outside the NIO box is rejected with
    OUT_OF_DOMAIN (rejected 422, never silently clamped) — the IMD/RSMC New
    Delhi basin the service is scoped to.
    """

    model_config = ConfigDict(extra="forbid")

    meta: Optional[AnalyzeMeta] = Field(
        default=None, description="optional system identity / attribution")

    @model_validator(mode="after")
    def _check_nio_domain(self) -> "AnalyzeRequest":
        lo_lat, hi_lat = NIO_LAT_RANGE
        lo_lon, hi_lon = NIO_LON_RANGE
        for i, obs in enumerate(self.history):
            lat, lon = obs.latitude, obs.longitude
            if not (lo_lat <= lat <= hi_lat and lo_lon <= lon <= hi_lon):
                raise ValidationCode(
                    OUT_OF_DOMAIN,
                    f"observation {i} ({lat:.4f}, {lon:.4f}) is outside the "
                    f"North Indian Ocean domain: lat in "
                    f"[{lo_lat}, {hi_lat}], lon in [{lo_lon}, {hi_lon}] "
                    f"(degrees East); rejected, not clamped")
        return self


# -- response blocks (mirror of the frontend contract) ---------------------

class LocationBlock(BaseModel):
    lat: float
    lon: float


class MetaBlock(BaseModel):
    systemId: str
    systemName: str
    basin: str
    lastPass: str
    source: str


class DetectionBlock(BaseModel):
    detected: bool
    confidence: int
    location: LocationBlock
    movementDirection: str
    movementSpeedKmh: float


class ClassificationBlock(BaseModel):
    category: str
    scale: str = "IMD"
    windSpeedKmh: float
    pressureHpa: float
    confidence: int
    structuralPattern: str


class ForecastEntry(BaseModel):
    hour: int
    label: str
    lat: float
    lon: float
    windSpeedKmh: float
    pressureHpa: Optional[float] = None
    confidence: Optional[int] = None


class LandfallBlock(BaseModel):
    estimated: bool
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    estimated_time: Optional[str] = None
    predictedWindKmh: Optional[float] = None
    distanceToLandKm: Optional[int] = None


class RiskBlock(BaseModel):
    score: int
    level: str


class HistoryTrackPoint(BaseModel):
    lat: float
    lon: float
    timestamp: str


class HistoryValue(BaseModel):
    t: str
    value: float


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    confidence: float


class SatelliteBlock(BaseModel):
    label: str
    boundingBox: Optional[BoundingBox] = None
    source: str


class ProvenanceBlock(BaseModel):
    pipeline: str
    sources: Dict[str, str]
    notes: List[str]
    reference_image: str
    tabular: Dict[str, Any]


class AnalyzeResponse(BaseModel):
    status: str = "success"
    meta: MetaBlock
    detection: DetectionBlock
    classification: ClassificationBlock
    forecast: List[ForecastEntry]
    landfall: LandfallBlock
    risk: RiskBlock
    historicalTrack: List[HistoryTrackPoint]
    windHistory: List[HistoryValue]
    pressureHistory: List[HistoryValue]
    confidenceHistory: List[HistoryValue]
    sstHistory: List[HistoryValue]
    envWindHistory: List[HistoryValue]
    satellite: SatelliteBlock
    provenance: ProvenanceBlock
    preprocessedPreview: Optional[str] = None


class ImageDetectionBlock(BaseModel):
    """P2 output on a user-uploaded image.

    location and movement are null: a single image provides presence/confidence
    and a pattern label, but NOT a geographic fix or motion (no validated
    localizer; movement requires a history).  We never invent a position.
    """

    detected: bool
    confidence: int
    location: Optional[LocationBlock] = None
    movementDirection: Optional[str] = None
    movementSpeedKmh: Optional[float] = None


class ImageClassificationBlock(BaseModel):
    """P3 category/confidence on a user-uploaded image.

    wind_speed/pressure from the image model are NOT exposed: the P3 wind
    regressor is degenerate and the image model's pressure is a hardcoded
    placeholder (990.0).  Only the softmax category + confidence are real.
    """

    category: str
    scale: str = "IMD"
    confidence: int
    structuralPattern: str


class ImageSatelliteBlock(BaseModel):
    """Source attribution for the analysed image.

    boundingBox is always null (no validated localizer in the artifact
    bundle); source carries ONLY the user-supplied original filename (never a
    filesystem path).
    """

    label: str
    boundingBox: Optional[BoundingBox] = None
    source: str


class ImageProvenanceBlock(BaseModel):
    """Honest provenance for /api/image. Note that we deliberately do NOT
    expose wind/pressure/landfall/risk for a user image alone: those depend on
    a validated 5-observation history, not on a single frame."""

    pipeline: str
    sources: Dict[str, str]
    notes: List[str]
    image_source: str
    tabular: Dict[str, Any]


class ImageAnalyzeResponse(BaseModel):
    """POST /api/image response — P2 + P3 on the user-uploaded image.

    Distinctly NOT a full dashboard payload: no forecast, no landfall, no risk,
    no history (those require the validated history via /api/analyze).  Honest
    nulls preserve the absence of a validated localizer and calibrated
    uncertainty.
    """

    status: str = "success"
    meta: MetaBlock
    detection: ImageDetectionBlock
    classification: ImageClassificationBlock
    satellite: ImageSatelliteBlock
    provenance: ImageProvenanceBlock
    preprocessedPreview: Optional[str] = None


class ForecastResponse(BaseModel):
    status: str = "success"
    model: Dict[str, Any]
    forecast: List[ForecastEntry]


class HealthBlock(BaseModel):
    status: str = "ok"
    service: str = "cyclone-integration"
    phase: str = "integration_api"
    offline: bool = True
    model_ready: bool
    ml: Dict[str, Any]
    forecasting: Dict[str, Any]