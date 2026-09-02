"""Private local geographic helpers (Haversine) reused from Phase-2 formula.

Phase-4 must not be coupled to Phase-2 code at test time when the Phase-2
package is unimportable, so the exact Phase-2 Haversine formula (IUGG mean
Earth radius, clipped) is mirrored here, byte-for-byte equivalent.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle (Haversine) distance in km, elementwise."""
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    phi1 = np.pi * lat1 / 180.0
    phi2 = np.pi * lat2 / 180.0
    dphi = np.pi * (lat2 - lat1) / 180.0
    dlambda = np.pi * (lon2 - lon1) / 180.0

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c