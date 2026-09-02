"""FastAPI application factory for the Phase-6 local forecasting API.

Local-only by design: binds to 127.0.0.1, CORS restricted to local dev
origins, fully offline (no network calls anywhere in the inference path).
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import (API_NAME, API_PHASE, CORS_ALLOW_CREDENTIALS,
                      CORS_ORIGINS, DEFAULT_HOST, DEFAULT_PORT,
                      MAX_REQUEST_BYTES)
from ..integration.forecasting_adapter import ForecastingAdapter
from .error_handlers import register_exception_handlers
from .routes import router

__all__ = ["create_app", "app"]


def create_app(adapter: Optional[ForecastingAdapter] = None,
               cors_origins=None) -> FastAPI:
    a = adapter if adapter is not None else ForecastingAdapter()

    app = FastAPI(
        title="Cyclone Forecasting API (P4 / SIH 2026)",
        version="1.0.0",
        description=(
            "Local inference API exposing the validated Phase-5 forecasting "
            "engine (EXP005, GRU+Huber). Offline; no external services. "
            "This is an integration layer and makes no model-performance "
            "claims beyond the audited Phase-4 results."),
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.adapter = a
    app.state.host = DEFAULT_HOST
    app.state.port = DEFAULT_PORT

    # CORS: local frontend development only (cwd Vite / CRA dev servers).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins if cors_origins is not None
                           else CORS_ORIGINS),
        allow_credentials=CORS_ALLOW_CREDENTIALS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def _body_size_guard(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_REQUEST_BYTES:
            return JSONResponse(
                {"status": "error",
                 "error": {"code": "INVALID_REQUEST",
                           "message": "request body exceeds the size limit"}},
                status_code=413)
        return await call_next(request)

    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()