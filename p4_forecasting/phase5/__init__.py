"""P4 Phase 5 — forecasting system integration & demo implementation.

Read-only inference/ integration layer around the audited Phase-4 champion
(EXP005).  Phase-5 never retrains and never modifies P1/Phase-1/Phase-2/
Phase-3/Phase-4 sources; it only reads them and writes under
``p4_forecasting/phase5/``.
"""

from . import config  # noqa: F401  (ensures repo path is importable)