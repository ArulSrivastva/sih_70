"""Causal derived-feature engineering for P4 Phase 4.

Locked contract
---------------
Raw history block ``X`` is ``(N, 5, 7)`` float32 with physical units::

    [lat, lon, wind_speed, pressure, sst, wind_u, wind_v]

The Phase-4 feature block is ``(N, 5, 16)``: the original seven features stay
first and byte-for-byte identical; nine causal derived features are appended,
computed per timestep using only the same or earlier information::

    7  delta_lat               12  pressure_change
    8  delta_lon               13  sst_change
    9  movement_speed          14  environmental_wind_speed
    10 movement_direction      15  environmental_wind_direction
    11 wind_change

Causality / first-step policy
-----------------------------
For the earliest history timestep (i = 0, t-24h) no predecessor exists inside
the model's causal input window. Per the locked Phase-4 decision, every
predecessor-dependent difference/trend feature is zero-filled at i = 0:

    "For difference/trend features requiring a predecessor, the first available
     history timestep is zero-filled because no predecessor exists within the
     model's causal input window."

The two environmental features are *not* predecessor-dependent: they are
computed from the current timestep's ``wind_u``/``wind_v`` at every timestep
including i = 0.  Target Y is never consulted; no future history step is used;
no interpolation is introduced.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..common import (
    DERIVED_FEATURES,
    FEATURE_NAMES,
    FEATURE_NAMES_RAW,
    HISTORY_STEPS,
    N_FEATURES,
)
from . import _geo  # private local haversine/bearing helpers (no phase2 dep)

RAW_NAMES = FEATURE_NAMES_RAW
DERIVED_NAMES = DERIVED_FEATURES
ALL_NAMES = FEATURE_NAMES

# History cadence: P1 contract is exactly 6-hourly.
SPACING_HOURS = 6.0

FIRST_STEP_ZERO_FILL_DOC = (
    "For difference/trend features requiring a predecessor, the first available "
    "history timestep is zero-filled because no predecessor exists within the "
    "model's causal input window."
)

# Indices (within the 7 raw columns) used by each predecessor-dependent feature.
LAT, LON, WIND, PRESSURE, SST, U, V = range(7)

# Column indices within the 16-feature output.
C_DLAT = 7
C_DLON = 8
C_MSPEED = 9
C_MDIR = 10
C_WCHG = 11
C_PCHG = 12
C_SCHG = 13
C_ESPEED = 14
C_EDIR = 15

# Predecessor-dependent derivative features (zero-filled at i=0).
PREDECESSOR_DEPENDENT = [
    "delta_lat", "delta_lon", "movement_speed", "movement_direction",
    "wind_change", "pressure_change", "sst_change",
]


def validate_raw(X: np.ndarray) -> np.ndarray:
    """Validate and coerce an (N,5,7) raw history block."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1:] != (HISTORY_STEPS, 7):
        raise ValueError(
            f"X must have shape (N,5,7); got {X.shape}")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN/Inf; refusing to engineer features")
    return X


def wrap_lon_delta(lon_prev: np.ndarray, lon_cur: np.ndarray) -> np.ndarray:
    """Wrapped longitudinal difference ``lon_cur - lon_prev`` in (-180, 180].

    Handles the 0/360 cyclicity of the canonical North-Indian-Ocean longitude
    convention so no artificial jump across the boundary is ever created.
    """
    lon_prev = np.asarray(lon_prev, dtype=np.float32)
    lon_cur = np.asarray(lon_cur, dtype=np.float32)
    return (lon_cur - lon_prev + 180.0) % 360.0 - 180.0


def bearing_degrees(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial great-circle bearing degrees clockwise from true north [0,360)."""
    lat1 = np.radians(np.asarray(lat1, dtype=np.float32))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float32))
    dlon = np.radians(wrap_lon_delta(lon1, lon2))
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brg = np.degrees(np.arctan2(y, x)) % 360.0
    return brg.astype(np.float32)


def environmental_wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological wind direction: the direction the wind blows FROM.

    Convention: degrees clockwise from true north in [0, 360).  u is the
    eastward 10-m component (m/s), v the northward 10-m component (m/s).

        from_dir = atan2(-u, -v) mod 360
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    return (np.degrees(np.arctan2(-u, -v)) % 360.0).astype(np.float32)


def _derived_at_step(X: np.ndarray, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 9 derived-value column-vectors at history step ``i``.

    Row ``i`` depends only on ``X[:, i, :]`` and (for i > 0) ``X[:, i-1, :]``.
    """
    cur = X[:, i, :]

    if i == 0:
        d_lat = np.zeros(cur.shape[0], dtype=np.float32)
        d_lon = np.zeros(cur.shape[0], dtype=np.float32)
        m_speed = np.zeros(cur.shape[0], dtype=np.float32)
        m_dir = np.zeros(cur.shape[0], dtype=np.float32)
        w_chg = np.zeros(cur.shape[0], dtype=np.float32)
        p_chg = np.zeros(cur.shape[0], dtype=np.float32)
        s_chg = np.zeros(cur.shape[0], dtype=np.float32)
    else:
        prev = X[:, i - 1, :]
        d_lat = (cur[:, LAT] - prev[:, LAT]).astype(np.float32)
        d_lon = wrap_lon_delta(prev[:, LON], cur[:, LON])
        m_speed = (_geo.haversine_km(prev[:, LAT], prev[:, LON], cur[:, LAT], cur[:, LON])
                   / SPACING_HOURS).astype(np.float32)
        m_dir = bearing_degrees(prev[:, LAT], prev[:, LON], cur[:, LAT], cur[:, LON])
        w_chg = (cur[:, WIND] - prev[:, WIND]).astype(np.float32)
        p_chg = (cur[:, PRESSURE] - prev[:, PRESSURE]).astype(np.float32)
        s_chg = (cur[:, SST] - prev[:, SST]).astype(np.float32)

    e_speed = np.hypot(cur[:, U], cur[:, V]).astype(np.float32)
    e_dir = environmental_wind_direction(cur[:, U], cur[:, V])
    return d_lat, d_lon, m_speed, m_dir, w_chg, p_chg, s_chg, e_speed, e_dir


def engineer_features(X: np.ndarray) -> np.ndarray:
    """Expand a raw ``(N,5,7)`` history block into ``(N,5,16)``.

    The first seven columns are byte-for-byte the input; the nine derived
    columns are causal (never use target values, future steps or i=-1).
    """
    X = validate_raw(X)
    N = X.shape[0]
    out = np.empty((N, HISTORY_STEPS, N_FEATURES), dtype=np.float32)
    out[:, :, :7] = X

    for i in range(HISTORY_STEPS):
        cols = _derived_at_step(X, i)
        for c, col in enumerate(cols, start=7):
            out[:, i, c] = col

    if np.isnan(out).any() or np.isinf(out).any():
        raise RuntimeError("feature engineering introduced NaN/Inf")
    return out


def features_from_history(history: np.ndarray) -> np.ndarray:
    """Convenience for a single (5,7) sample -> (5,16) feature block."""
    h = np.asarray(history, dtype=np.float32)
    if h.ndim != 2 or h.shape != (HISTORY_STEPS, 7):
        raise ValueError(f"history must be (5,7); got {h.shape}")
    return engineer_features(h[None, ...])[0]