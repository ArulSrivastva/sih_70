"""Geographic metrics for cyclone forecasting (P4 Phase 2).

Track error is always computed on the sphere using the Haversine formula
(never Euclidean degree distance).
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Great-circle (Haversine) distance in kilometres.

    Args:
        lat1, lon1: first point(s), degrees (scalar or array)
        lat2, lon2: second point(s), degrees (scalar or array)

    Returns:
        distance in km, same shape/broadcast as inputs.
    """
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)

    if lon1.shape != lon2.shape or lat1.shape != lat2.shape:
        np.broadcast_arrays(lat1, lon1, lat2, lon2)  # raises if non-compatible

    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    dist = EARTH_RADIUS_KM * c
    if np.isscalar(lat1) and np.isscalar(lon1) and np.isscalar(lat2) and np.isscalar(lon2):
        return float(dist)
    return dist