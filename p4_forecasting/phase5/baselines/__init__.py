"""Phase-5 baselines: reference forecasts for demo / comparison.

Both modules reproduce the audited Phase-2 definitions exactly (persistence
and movement-vector) and are parity-tested against ``phase2.baselines``.
"""

from .persistence import persistence_forecast, PersistenceBaseline
from .movement_vector import (movement_vector_forecast, MovementVectorBaseline)

__all__ = [
    "persistence_forecast", "PersistenceBaseline",
    "movement_vector_forecast", "MovementVectorBaseline",
]