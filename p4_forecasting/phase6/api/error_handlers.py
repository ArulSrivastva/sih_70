"""Structured error handling.  API users NEVER see stack traces or filesystem
paths; details are only logged server-side."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Tuple

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..integration.forecasting_adapter import AdapterError
from ..schemas.requests import ValidationCode

logger = logging.getLogger("phase6.api")


def _body(code: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "error": {"code": code, "message": message}}


def _first_problem(exc: RequestValidationError) -> Tuple[str, str, bool]:
    """(code, message, suppress_input) from the first pydantic problem."""
    errors = exc.errors()
    if not errors:
        return "INVALID_REQUEST", "invalid request data", True
    err = errors[0]
    ctx = err.get("ctx") or {}
    marker = ctx.get("error")
    if isinstance(marker, ValidationCode):
        return marker.code, marker.message, False
    t = str(err.get("type", ""))
    loc = ".".join(str(x) for x in err.get("loc", ()))
    if t in ("too_short", "too_long"):
        return ("INVALID_HISTORY_LENGTH",
                f"history must contain exactly 5 observations at "
                f"t-24h/t-18h/t-12h/t-6h/t; problem near '{loc}'", True)
    if t == "finite_number":
        return "NON_FINITE_VALUE", \
            f"history contains NaN or infinite values; problem near '{loc}'", True
    if t == "missing":
        locs = [str(x) for x in err.get("loc", ())]
        if "history" in locs:
            return "MISSING_FEATURE", \
                f"missing required feature near '{loc}'", True
        return "INVALID_REQUEST", \
            "request body missing or not a valid forecast request", True
    if t == "json_invalid":
        return "INVALID_REQUEST", "Malformed JSON body", True
    return "INVALID_REQUEST", str(err.get("msg", "invalid request data")), True


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _on_validation(request: Request, exc: RequestValidationError):
        code, message, _ = _first_problem(exc)
        logger.info("validation rejected %s %s: %s [%s]",
                    request.method, request.url.path, code, message)
        return JSONResponse(_body(code, message), status_code=422)

    @app.exception_handler(AdapterError)
    async def _on_adapter(request: Request, exc: AdapterError):
        logger.warning("adapter error on %s %s: %s [%s]",
                       request.method, request.url.path, exc.code, exc.message)
        return JSONResponse(_body(exc.code, exc.message),
                            status_code=exc.http_status)

    @app.exception_handler(StarletteHTTPException)
    async def _on_http(request: Request, exc: StarletteHTTPException):
        return JSONResponse(_body("INVALID_REQUEST", str(exc.detail)),
                            status_code=exc.status_code)

    @app.exception_handler(json.JSONDecodeError)
    async def _on_bad_json(request: Request, exc: json.JSONDecodeError):
        logger.info("malformed JSON on %s %s", request.method,
                    request.url.path)
        return JSONResponse(_body("INVALID_REQUEST", "Malformed JSON body"),
                            status_code=422)

    @app.exception_handler(Exception)
    async def _on_generic(request: Request, exc: Exception):
        logger.exception("unhandled error on %s %s",
                         request.method, request.url.path)
        return JSONResponse(_body("INFERENCE_ERROR",
                                  "internal error; see server log"),
                            status_code=500)