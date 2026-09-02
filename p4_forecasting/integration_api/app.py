"""integration_api / app.py

Composes the audited Phase-6 FastAPI app with the ``/api`` integration
router.  Reusing the phase-6 factory keeps CORS, the request-size guard and
the structured error handlers identical; ``app.state.analyzer`` is the only
new piece of state.

Start (from p4_forecasting/)::

    python -X utf8 -m uvicorn integration_api.app:app \
        --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from phase6.integration.forecasting_adapter import ForecastingAdapter

from .analyzer import CycloneAnalyzer
from .routes import router


def create_app(adapter: Optional[ForecastingAdapter] = None,
               cors_origins=None) -> FastAPI:
    from phase6.api.app import create_app as _create_phase6_app

    base = _create_phase6_app(adapter=adapter, cors_origins=cors_origins)
    base.state.analyzer = CycloneAnalyzer(base.state.adapter)
    base.include_router(router)
    return base


app = create_app()