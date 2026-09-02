"""Source immutability: exercising the API must not modify P1 / Phase-1/2/3/4/5
or anything outside phase6."""

import hashlib
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent.parent

SAMPLE_SOURCES = [
    "p4_forecasting/phase5/service/forecasting_service.py",
    "p4_forecasting/phase5/inference/predictor.py",
    "p4_forecasting/phase5/inference/preprocessing.py",
    "p4_forecasting/phase5/inference/input_validation.py",
    "p4_forecasting/phase5/baselines/persistence.py",
    "p4_forecasting/phase5/baselines/movement_vector.py",
    "p4_forecasting/phase5/config.py",
    "p4_forecasting/phase5/run_phase5.py",
    "p4_forecasting/phase4/results/normalization_stats.json",
    "p4_forecasting/phase4/results/champion_model.json",
    "p4_forecasting/phase4/results/experiments/EXP005/config.json",
    "p4_forecasting/phase4/results/experiments/EXP005/checkpoint.pt",
    "p4_forecasting/phase4/features/feature_engineering.py",
    "p4_forecasting/phase3/results/model_comparison.json",
    "p4_forecasting/phase2/results/baseline_results.json",
]


def _sha(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _snapshot():
    return {rel: _sha(BASE / rel) for rel in SAMPLE_SOURCES
            if (BASE / rel).exists()}


def test_api_calls_do_not_modify_sources(client, valid_request):
    before = _snapshot()
    client.get("/health")
    client.get("/model")
    client.post("/forecast", json=valid_request)
    client.post("/forecast/compare", json=valid_request)
    client.get("/openapi.json")
    after = _snapshot()
    assert before == after


def test_no_files_created_outside_phase6_during_calls(client,
                                                      valid_request,
                                                      tmp_path):
    """No __pycache__ / .pytest_cache / junk files appear outside phase6."""
    import os
    scan_before = set()
    for dirpath, dirnames, filenames in os.walk(BASE.parent):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".pytest_cache")]
        for name in filenames:
            scan_before.add(Path(dirpath) / name)
    client.post("/forecast", json=valid_request)
    client.post("/forecast/compare", json=valid_request)
    scan_after = set()
    for dirpath, dirnames, filenames in os.walk(BASE.parent):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".pytest_cache")]
        for name in filenames:
            scan_after.add(Path(dirpath) / name)
    new_files = {str(p) for p in scan_after - scan_before
                 if "phase6" not in p.as_posix().replace("\\", "/")}
    assert not new_files, new_files