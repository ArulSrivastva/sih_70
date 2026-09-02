"""Shared pytest fixtures for P4 Phase-4 tests.

All tests are self-contained / synthetic.  Anything a test writes goes into a
scratch dir under ``p4_forecasting/phase4/tests/.scratch/`` (NOT pytest tmp_path
and NOT any Read-only Phase-1/2/3 artifact directory).  The scratch folder is
deleted after the run.
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

PKG_DIR = Path(__file__).resolve().parents[2]  # p4_forecasting
PHASE4_DIR = PKG_DIR / "phase4"

for _p in (str(PKG_DIR), str(PHASE4_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

TEST_ROOT = Path(__file__).resolve().parent
SCRATCH_ROOT = TEST_ROOT / ".scratch"


@pytest.fixture(scope="session", autouse=True)
def _clean_scratch_root():
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


@pytest.fixture()
def scratch(tmp_path_factory):
    d = SCRATCH_ROOT / str(uuid.uuid4())
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def synthetic_raw_history():
    """Deterministic (16, 5, 7) raw history block for feature tests.

    Uses plausible lat/lon/wind/pressure/SST/u/v values with a clean 0/360
    longitude crossing in the middle.
    """
    rng = np.random.RandomState(7)
    X = rng.uniform(size=(16, 5, 7)).astype(np.float32)
    X[:, :, 0] = np.linspace(8.0, 22.0, 5, dtype=np.float32)      # lat
    X[:, :, 1] = np.array([60.0, 70.0, 359.0, 1.0, 5.0], dtype=np.float32)  # lon (+wrap)
    X[:, :, 2] = np.array([60.0, 80.0, 100.0, 120.0, 140.0], dtype=np.float32)  # wind
    X[:, :, 3] = np.array([980.0, 975.0, 970.0, 965.0, 960.0], dtype=np.float32)  # pressure
    X[:, :, 4] = 29.5                                                        # sst
    X[:, :, 5] = np.array([5.0, 6.0, -3.0, 4.0, 7.0], dtype=np.float32)   # u
    X[:, :, 6] = np.array([2.0, -1.0, 4.0, 3.0, -5.0], dtype=np.float32)  # v
    return X


def make_feature_npz(path: Path, n_train: int = 48, n_val: int = 12,
                     seed: int = 42) -> Path:
    """Write a tiny synthetic (5,16)/(3,3) feature dataset split pair."""
    from phase4.common import FEATURE_NAMES, TARGET_NAMES
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    for split, n in (("train", n_train), ("val", n_val), ("test", 8)):
        X = rng.uniform(size=(n, 5, 16)).astype(np.float32)
        Y = rng.uniform(size=(n, 3, 3)).astype(np.float32)
        np.savez_compressed(
            path / f"{split}.npz", X=X, Y=Y,
            features=np.array(FEATURE_NAMES), targets=np.array(TARGET_NAMES),
            horizons=np.array([6, 12, 24]))
        import pandas as pd
        pd.DataFrame({
            "cyclone_id": [f"C-{i}" for i in range(n)],
            "t_zero": [f"2023-01-{i % 28 + 1:02d}T00:00:00" for i in range(n)],
            "source_split": [split] * n,
            "original_sample_index": list(range(n)),
        }).to_csv(path / f"{split}_metadata.csv", index=False)
    return path


def make_normalization_stats(n: int = 48, seed: int = 0) -> dict:
    from phase4.training.normalization import compute_normalization_stats
    rng = np.random.RandomState(seed)
    X = rng.uniform(size=(n, 5, 16)).astype(np.float32)
    Y = rng.uniform(size=(n, 3, 3)).astype(np.float32)
    return compute_normalization_stats(X, Y)