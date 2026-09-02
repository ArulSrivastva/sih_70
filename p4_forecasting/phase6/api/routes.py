"""HTTP routes: /health, /model, /forecast, /forecast/compare."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..integration.forecasting_adapter import AdapterError, ForecastingAdapter
from ..schemas.requests import ForecastRequest, CompareRequest
from ..schemas.responses import (CompareSuccessResponse, ForecastItem,
                                 ForecastModelBlock, ForecastSuccessResponse,
                                 HealthResponse, ModelInfoResponse)

router = APIRouter()


def _adapter(request: Request) -> ForecastingAdapter:
    return request.app.state.adapter


@router.get("/health", response_model=HealthResponse,
            summary="Service health and model readiness")
def health(request: Request) -> HealthResponse:
    return HealthResponse(**_adapter(request).health())


@router.get("/model", response_model=ModelInfoResponse,
            summary="Model metadata read from audited artifacts")
def model_info(request: Request) -> ModelInfoResponse:
    info = _adapter(request).model_info()
    return ModelInfoResponse(**info)


@router.post("/forecast", response_model=ForecastSuccessResponse,
             summary="Forecast +6h/+12h/+24h from 24h of history")
def forecast(request: Request, body: ForecastRequest) -> ForecastSuccessResponse:
    res = _adapter(request).forecast(body)
    return ForecastSuccessResponse(
        status="success",
        model=ForecastModelBlock.from_phase5(res["model"]),
        input=res["input"],
        forecast=[ForecastItem(**f) for f in res["forecast"]],
    )


@router.post("/forecast/compare", response_model=CompareSuccessResponse,
             summary="Model forecast alongside persistence and movement-vector")
def forecast_compare(request: Request,
                     body: CompareRequest) -> CompareSuccessResponse:
    res = _adapter(request).compare(body)
    return CompareSuccessResponse.from_phase5(res)