"""Phase-5 pytest configuration.

Deliberately write-free: unlike Phase-4's conftest, this suite must not create
a .scratch dir or touch anything outside ``p4_forecasting/phase5``.  Bytecode
writing is disabled so no __pycache__ is produced anywhere.
"""

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import pathlib

_PHASE5 = pathlib.Path(__file__).resolve().parent.parent
_PKG = _PHASE5.parent
for _p in (_PKG, str(_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest


@pytest.fixture(scope="session")
def example_history_dict():
    return {
        "timestamps": [
            "2024-08-25T00:00:00Z", "2024-08-25T06:00:00Z",
            "2024-08-25T12:00:00Z", "2024-08-25T18:00:00Z",
            "2024-08-26T00:00:00Z",
        ],
        "history": [
            {"lat": 23.30, "lon": 68.50, "wind_speed": 55.6, "pressure": 994.0,
             "sst": 28.67, "wind_u": 8.5095, "wind_v": 1.2749},
            {"lat": 23.40, "lon": 68.00, "wind_speed": 64.8, "pressure": 993.0,
             "sst": 28.26, "wind_u": 4.4456, "wind_v": 7.0864},
            {"lat": 23.40, "lon": 67.10, "wind_speed": 64.8, "pressure": 993.0,
             "sst": 28.43, "wind_u": 7.9810, "wind_v": -3.3996},
            {"lat": 23.60, "lon": 66.40, "wind_speed": 64.8, "pressure": 992.0,
             "sst": 28.69, "wind_u": 2.0987, "wind_v": -4.1294},
            {"lat": 23.50, "lon": 65.70, "wind_speed": 74.1, "pressure": 991.0,
             "sst": 28.85, "wind_u": 2.4551, "wind_v": 4.3582},
        ],
    }


@pytest.fixture(scope="session")
def example_history_np(example_history_dict):
    import numpy as np
    steps = example_history_dict["history"]
    fields = ["lat", "lon", "wind_speed", "pressure", "sst", "wind_u", "wind_v"]
    return np.array([[s[f] for f in fields] for s in steps], dtype=np.float32)


@pytest.fixture(scope="session")
def service():
    from phase5.service.forecasting_service import ForecastingService
    return ForecastingService()