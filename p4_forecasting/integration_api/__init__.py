"""Integration API for the SIH 2026 cyclone-project.

Wires the audited Phase-4/5 forecasting engine (via ``phase6``), the
Phase-2 detection model and the Phase-3 classification model into a single
local ``/api`` orchestrator that speaks the dashboard contract exactly.

Read-only by design:
  * P2/P3 weights and the reference INSAT frame are served from
    ``PS70-main.zip`` entirely in memory (``zipfile`` + ``BytesIO``).
    Nothing is ever extracted to disk.
  * No network module is imported anywhere in the inference path and the
    torch backbones are built with ``weights=None`` so no pretrained weight
    download can occur.
  * No accuracy claim is made beyond the audited Phase-4 results; each
    response block carries provenance describing its real source.
"""

__version__ = "1.0.0"