"""Response models (the stable JSON the frontend consumes)."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "cyclone-forecasting"
    phase: str = "phase6"
    offline: bool = True
    model_ready: bool = Field(
        ..., description="true when the local champion artifacts exist "
        "and can be loaded")


class ForecastItem(BaseModel):
    hours: int
    latitude: float
    longitude: float
    wind_speed_kmh: float

    @classmethod
    def from_list(cls, entries: List[Dict[str, Any]]) -> List[Any]:
        return [cls(**e) for e in entries]


class ForecastModelBlock(BaseModel):
    experiment_id: str
    family: str
    loss: str

    @classmethod
    def from_phase5(cls, model: Dict[str, Any]) -> "ForecastModelBlock":
        return cls(experiment_id=model["experiment_id"],
                   family=model["family"], loss=model["loss"])


class ForecastSuccessResponse(BaseModel):
    status: str = "success"
    model: ForecastModelBlock
    input: Dict[str, int]
    forecast: List[ForecastItem]


class CompareSuccessResponse(BaseModel):
    status: str = "success"
    model: ForecastModelBlock
    model_forecast: List[ForecastItem]
    persistence_forecast: List[ForecastItem]
    movement_vector_forecast: List[ForecastItem]

    @classmethod
    def from_phase5(cls, payload: Dict[str, Any]) -> "CompareSuccessResponse":
        return cls(
            status="success",
            model=ForecastModelBlock.from_phase5(payload["model"]),
            model_forecast=ForecastItem.from_list(payload["model_forecast"]),
            persistence_forecast=ForecastItem.from_list(
                payload["persistence_forecast"]),
            movement_vector_forecast=ForecastItem.from_list(
                payload["movement_vector_forecast"]),
        )


class ModelInfoResponse(BaseModel):
    experiment_id: str
    model: str
    loss: str
    hidden_size: int
    layers: int
    input_size: int
    output_size: int
    history_steps: int
    history_hours: int
    feature_count: int
    horizons: List[int]
    targets: List[str]
    validation_primary_score: float


class ApiErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    status: str = "error"
    error: ApiErrorBody