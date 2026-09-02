"""Pytest config for P4 Phase 3 tests.

Ensures `p4_forecasting` is importable (needed to reuse phase2 evaluation code).
"""

from __future__ import annotations

import sys
from pathlib import Path

P4_DIR = Path(__file__).resolve().parents[2]  # .../p4_forecasting
if str(P4_DIR) not in sys.path:
    sys.path.insert(0, str(P4_DIR))