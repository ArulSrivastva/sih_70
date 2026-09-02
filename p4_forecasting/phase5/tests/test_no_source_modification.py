"""Safety: no Phase-1/2/3/4/source files modified by the Phase-5 code paths."""

import hashlib
from pathlib import Path

import numpy as np

from phase5.service.forecasting_service import ForecastingService
from phase5.config import default_paths


def _sha(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


SAMPLE_SOURCES = [
    "p4_forecasting/phase4/results/normalization_stats.json",
    "p4_forecasting/phase4/results/champion_model.json",
    "p4_forecasting/phase4/results/FINAL_COMPARISON.json",
    "p4_forecasting/phase4/results/experiments/EXP005/config.json",
    "p4_forecasting/phase4/results/experiments/EXP005/checkpoint.pt",
    "p4_forecasting/phase4/features/feature_engineering.py",
    "p4_forecasting/phase4/training/normalization.py",
    "p4_forecasting/phase4/models/gru.py",
    "p4_forecasting/phase2/results/baseline_results.json",
    "p4_forecasting/phase3/results/model_comparison.json",
]


def _snapshot():
    return {rel: _sha(Path(rel)) for rel in SAMPLE_SOURCES
            if Path(rel).exists()}


def test_phase4_and_earlier_sources_unchanged_by_phase5(service,
                                                        example_history_dict):
    before = _snapshot()
    paths = default_paths()
    # exercise every Phase-5 read path
    service.forecast(example_history_dict)
    service.compare_baselines(example_history_dict)
    service.forecast({k: example_history_dict[k] for k in
                      ("history", "timestamps")})
    from phase5.baselines.persistence import persistence_forecast
    from phase5.baselines.movement_vector import movement_vector_forecast
    import json
    h7 = np.array([[s["lat"], s["lon"], s["wind_speed"], s["pressure"],
                    s["sst"], s["wind_u"], s["wind_v"]]
                   for s in example_history_dict["history"]], np.float32)
    persistence_forecast(h7)
    movement_vector_forecast(h7)
    _ = json.loads(paths.champion_meta.read_text(encoding="utf-8"))
    after = _snapshot()
    assert before == after, f"source files changed: {after}"


def test_no_phase5_writes_outside_phase5():
    """Phase-5 inference/baseline/service modules must not open any file for
    writing (AST-level scan of every package .py under phase5)."""
    import ast
    from pathlib import Path
    base = Path("p4_forecasting/phase5")
    offenders = []
    for py in sorted(base.rglob("*.py")):
        if py.name in ("run_phase5.py", "snapshot.py"):
            continue  # the runner and snapshot tool are the ONLY writers
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (getattr(fn, "id", None) or
                        (getattr(fn, "attr", None) if isinstance(fn, ast.Attribute)
                         else None))
                if name != "open":
                    continue
                mode = None
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                if isinstance(mode, str) and mode[:1] in "wax":
                    offenders.append(f"{py}: open(mode={mode!r})")
    assert not offenders, offenders