"""ForecastingAdapter: the single seam between the Phase-6 HTTP layer and
the validated Phase-5 ``ForecastingService``.

The adapter owns NO scientific logic.  It only:
  * maps the Phase-6 request schema onto the Phase-5 input contract,
  * calls ``service.forecast`` / ``service.compare_baselines``,
  * maps Phase-5 error codes onto the public Phase-6 error vocabulary,
  * reads model metadata from the audited artifacts.

Documented Phase-5 integration path (inspected before coding):

    phase5/service/forecasting_service.py  ->  ForecastingService
        .forecast(history)            dict-form/numpy history -> response dict
        .compare_baselines(history)   response dict with the three forecasts
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

_PKG = Path(__file__).resolve().parent.parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from phase5.service.forecasting_service import ForecastingService  # noqa: E402

from ..config import (API_NAME, API_PHASE, PHASE5_TO_PHASE6_CODE,  # noqa: E402
                      champion_identity, default_paths)
from ..schemas.requests import ForecastRequest  # noqa: E402


class AdapterError(Exception):
    """Raised by the adapter for structured API-level failures."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = _status_for_code(code)


def _status_for_code(code: str) -> int:
    if code == "MODEL_NOT_READY":
        return 503
    if code == "INFERENCE_ERROR":
        return 500
    return 422


class ForecastingAdapter:
    """Wraps the Phase-5 service; lazy predictor loading (cold load happens
    on first inference)."""

    def __init__(self, service: Optional[ForecastingService] = None) -> None:
        self._service = service if service is not None else ForecastingService()
        self._cold_load_ms: Union[float, None] = None
        self._inference_calls = 0

    # -- health -------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        missing = default_paths().require()
        ready = not missing
        return {
            "status": "ok",
            "service": API_NAME,
            "phase": API_PHASE,
            "offline": True,
            "model_ready": ready,
        }

    # -- model metadata ------------------------------------------------------
    def model_info(self) -> Dict[str, Any]:
        try:
            info = champion_identity()
            family = {"gru": "GRU", "improved_lstm": "ImprovedLSTM",
                      "multitask_lstm": "MultiTaskLSTM"}.get(
                str(info["model"]).lower(), str(info["model"]))
            return {**info,
                    "model": family,
                    "loss": str(info["loss"]).title(),
                    "validation_primary_score": float(
                        info["validation_primary_score"] or 0.0)}
        except Exception as exc:
            raise AdapterError("MODEL_NOT_READY",
                               f"model metadata unavailable: {exc}")

    # -- inference -----------------------------------------------------------
    def forecast(self, request: ForecastRequest) -> Dict[str, Any]:
        """Validated request -> Phase-5 forecast response dict."""
        self._inference_calls += 1
        try:
            res = self._service.forecast(request.phase5_document())
        except AdapterError:
            raise
        except Exception:
            raise AdapterError(
                "INFERENCE_ERROR",
                "forecasting service failed internally; see server log")
        return self._unwrap(res)

    def compare(self, request: ForecastRequest) -> Dict[str, Any]:
        """Validated request -> Phase-5 comparative response dict."""
        self._inference_calls += 1
        try:
            res = self._service.compare_baselines(request.phase5_document())
        except Exception:
            raise AdapterError(
                "INFERENCE_ERROR",
                "forecasting service failed internally; see server log")
        return self._unwrap(res)

    def _unwrap(self, res: Dict[str, Any]) -> Dict[str, Any]:
        if res.get("status") == "success":
            return res
        body = res.get("error", {}) if isinstance(res, dict) else {}
        code = PHASE5_TO_PHASE6_CODE.get(str(body.get("code")),
                                         "INFERENCE_ERROR")
        raise AdapterError(code, str(body.get(
            "message", "forecasting service rejected the input")))

    # -- cold-load / latency helpers ------------------------------------------
    def ensure_loaded(self) -> float:
        """Force the predictor to load now; returns the cold-load wall time."""
        import time
        t0 = time.perf_counter()
        _ = self._service.predictor
        self._cold_load_ms = (time.perf_counter() - t0) * 1000.0
        return self._cold_load_ms

    @property
    def cold_load_ms(self) -> Union[float, None]:
        return self._cold_load_ms

    @property
    def service(self) -> ForecastingService:
        return self._service

    @property
    def inference_calls(self) -> int:
        return self._inference_calls