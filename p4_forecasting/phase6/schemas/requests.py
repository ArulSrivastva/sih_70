"""Request models with strict, Phase-6 level validation.

All validation happens HERE, before the forecasting service is called.
Invalid input is rejected (HTTP 422), never silently repaired, interpolated
or guessed.  The service only ever sees fully-validated history.
"""

from __future__ import annotations

import datetime as _dt
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import (HISTORY_STEPS, LAT_RANGE, LON_RANGE, PRESSURE_RANGE,
                      SST_RANGE, SPACING_HOURS, WIND_RANGE)


class ValidationCode(ValueError):
    """Raised inside validators to carry an exact public error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ForecastObservation(BaseModel):
    """One 6-hourly cyclone observation (t-24h .. t)."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    latitude: float = Field(..., allow_inf_nan=False)
    longitude: float = Field(..., allow_inf_nan=False)
    wind_speed_kmh: float = Field(..., allow_inf_nan=False)
    pressure_hpa: float = Field(..., allow_inf_nan=False)
    sst: float = Field(..., allow_inf_nan=False)
    wind_u: float = Field(..., allow_inf_nan=False)
    wind_v: float = Field(..., allow_inf_nan=False)

    @field_validator("latitude")
    @classmethod
    def _check_latitude(cls, v: float) -> float:
        lo, hi = LAT_RANGE
        if not (lo <= v <= hi):
            raise ValidationCode(
                "INVALID_LATITUDE",
                f"latitude must be within [{lo}, {hi}]; got {v}")
        return v

    @field_validator("longitude")
    @classmethod
    def _check_longitude(cls, v: float) -> float:
        lo, hi = LON_RANGE
        if not (lo <= v < hi):
            raise ValidationCode(
                "INVALID_LONGITUDE",
                f"longitude must be in [{lo}, {hi}) (canonical 0..360 "
                f"degrees-east); got {v}")
        return v

    @field_validator("wind_speed_kmh")
    @classmethod
    def _check_wind(cls, v: float) -> float:
        lo, hi = WIND_RANGE
        if not (lo <= v <= hi):
            raise ValidationCode(
                "INVALID_WIND",
                f"wind_speed_kmh must be within [{lo}, {hi}]; got {v}")
        return v

    @field_validator("pressure_hpa")
    @classmethod
    def _check_pressure(cls, v: float) -> float:
        lo, hi = PRESSURE_RANGE
        if not (lo <= v <= hi):
            raise ValidationCode(
                "INVALID_REQUEST",
                f"pressure_hpa outside physically-plausible range "
                f"[{lo}, {hi}]; got {v}")
        return v

    @field_validator("sst")
    @classmethod
    def _check_sst(cls, v: float) -> float:
        lo, hi = SST_RANGE
        if not (lo <= v <= hi):
            raise ValidationCode(
                "INVALID_REQUEST",
                f"sst outside physically-plausible range [{lo}, {hi}]; "
                f"got {v}")
        return v

    def phase5_step(self) -> dict:
        """Map this validated observation to the Phase-5 dict form."""
        return {
            "lat": self.latitude,
            "lon": self.longitude,
            "wind_speed": self.wind_speed_kmh,
            "pressure": self.pressure_hpa,
            "sst": self.sst,
            "wind_u": self.wind_u,
            "wind_v": self.wind_v,
        }


class ForecastRequest(BaseModel):
    """Exactly 5 observations at t-24h / t-18h / t-12h / t-6h / t."""

    model_config = ConfigDict(extra="forbid")

    history: List[ForecastObservation] = Field(
        ..., min_length=HISTORY_STEPS, max_length=HISTORY_STEPS,
        description=f"exactly {HISTORY_STEPS} 6-hourly observations")

    @model_validator(mode="after")
    def _check_timestamps(self) -> "ForecastRequest":
        parsed: List[_dt.datetime] = []
        for obs in self.history:
            try:
                parsed.append(_dt.datetime.fromisoformat(obs.timestamp))
            except ValueError:
                raise ValidationCode(
                    "INVALID_TIMESTAMP",
                    f"cannot parse timestamp {obs.timestamp!r} "
                    f"(expected ISO-8601, e.g. 2025-11-29T00:00:00Z)")
        for i in range(1, len(parsed)):
            gap = parsed[i] - parsed[i - 1]
            if gap <= _dt.timedelta(0):
                raise ValidationCode(
                    "NON_MONOTONIC_HISTORY",
                    "history timestamps must be strictly increasing")
            if gap != _dt.timedelta(hours=SPACING_HOURS):
                raise ValidationCode(
                    "INVALID_HISTORY_SPACING",
                    f"history observations must be exactly "
                    f"{SPACING_HOURS} hours apart; found {gap}")
        return self

    def phase5_document(self) -> dict:
        """The (already fully validated) request as the Phase-5 input dict."""
        return {
            "timestamps": [o.timestamp for o in self.history],
            "history": [o.phase5_step() for o in self.history],
        }


class CompareRequest(ForecastRequest):
    """Identical input shape for the baseline-comparison endpoint."""