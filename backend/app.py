"""backend / app.py

Top-level FastAPI application entrypoint for the Cyclone Detection,
Classification & Forecasting pipeline.

Delegates directly to the audited integration_api application.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P4_DIR = ROOT / "p4_forecasting"
if str(P4_DIR) not in sys.path:
    sys.path.insert(0, str(P4_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration_api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
