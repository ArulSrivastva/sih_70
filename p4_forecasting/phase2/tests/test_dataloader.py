""". Tests for the Phase-2 DataLoader against the CLEAN-only chronological dataset.

Standalone runnable:  python -m pytest p4_forecasting/phase2/tests/test_dataloader.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ..dataloader import build_clean_dataset
from ..dataloader.build_dataloaders import build_dataloaders
from ..dataloader.forecasting_dataset import (
    FEATURES,
    TARGETS,
    CycloneForecastingDataset,
    collate_forecasting,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
P4_ROOT = PROJECT_ROOT / "p4_forecasting"
PHASE2_ROOT = P4_ROOT / "phase2"
CHRONO_DIR = P4_ROOT / "canonical_chrono"
QUALITY_CSV = P4_ROOT / "canonical" / "sample_quality.csv"
DATASET_DIR = PHASE2_ROOT / "results" / "canonical_chronological_clean"

ORIGIN_TOL = 1e-4  # float32 tolerance for last-history-step == origin


def _ensure_clean_dataset() -> None:
    """Build the CLEAN-only dataset in-place if it is not present (idempotent)."""
    if not (DATASET_DIR / "train.npz").exists():
        build_clean_dataset.build_clean_chronological_dataset(CHRONO_DIR, QUALITY_CSV, DATASET_DIR)


@pytest.fixture(scope="module")
def clean_data():
    _ensure_clean_dataset()
    return build_clean_dataset.load_clean_splits(DATASET_DIR)


def test_clean_counts_train(clean_data):
    assert clean_data["train"]["X"].shape[0] == 1212
    assert clean_data["train"]["meta"]["cyclone_id"].nunique() == 57


def test_clean_counts_val(clean_data):
    assert clean_data["val"]["X"].shape[0] == 231
    assert clean_data["val"]["meta"]["cyclone_id"].nunique() == 13


def test_clean_counts_test(clean_data):
    assert clean_data["test"]["X"].shape[0] == 198
    assert clean_data["test"]["meta"]["cyclone_id"].nunique() == 10


def test_shapes(clean_data):
    for s in ("train", "val", "test"):
        X = clean_data[s]["X"]
        Y = clean_data[s]["Y"]
        assert X.shape[1:] == (5, 7), f"{s} X wrong: {X.shape}"
        assert Y.shape[1:] == (3, 3), f"{s} Y wrong: {Y.shape}"
        assert X.dtype == np.float32, f"{s} X dtype {X.dtype}"
        assert Y.dtype == np.float32, f"{s} Y dtype {Y.dtype}"


def test_no_nan_inf(clean_data):
    for s in ("train", "val", "test"):
        X = clean_data[s]["X"]
        Y = clean_data[s]["Y"]
        assert not np.isnan(X).any(), f"{s} X has NaN"
        assert not np.isinf(X).any(), f"{s} X has Inf"
        assert not np.isnan(Y).any(), f"{s} Y has NaN"
        assert not np.isinf(Y).any(), f"{s} Y has Inf"


def test_feature_order(clean_data):
    for s in ("train", "val", "test"):
        assert clean_data[s]["features"] == FEATURES
        assert clean_data[s]["targets"] == TARGETS


def test_last_history_equals_origin(clean_data):
    """Last history timestep must equal the origin/current observation (lat, lon, wind)."""
    for s in ("train", "val", "test"):
        X = clean_data[s]["X"]
        meta = clean_data[s]["meta"]
        origin = meta[["origin_lat", "origin_lon", "origin_wind"]].to_numpy(np.float32)
        last = np.asarray(X[:, -1, :3], dtype=np.float32)
        diff = np.abs(last - origin)
        assert float(diff.max()) < ORIGIN_TOL, f"{s} max |last-origin| = {float(diff.max())}"


def test_dataset_and_loaders():
    _ensure_clean_dataset()
    loaders = build_dataloaders(DATASET_DIR, batch_size=64, num_workers=0)
    counts = {"train": 1212, "val": 231, "test": 198}

    for s, loader in loaders.items():
        batch_iter = iter(loader)
        batch = next(batch_iter)
        b, h, t = batch["history"].shape[0], batch["history"].shape[1], batch["history"].shape[2]
        assert (h, t) == (5, 7), f"{s} history shape {batch['history'].shape}"
        assert batch["target"].shape[1:] == (3, 3)
        assert batch["history"].dtype == torch.float32
        assert batch["target"].dtype == torch.float32
        assert isinstance(batch["metadata"], list)

        n_seen = 0
        for batch in loader:
            n_seen += batch["history"].shape[0]
        assert n_seen == counts[s], f"{s} total samples {n_seen} != {counts[s]}"


def test_train_shuffled_val_test_not():
    _ensure_clean_dataset()
    from torch.utils.data.sampler import RandomSampler, SequentialSampler
    loaders = build_dataloaders(DATASET_DIR, batch_size=64, num_workers=0)
    assert isinstance(loaders["train"].sampler, RandomSampler)
    assert isinstance(loaders["val"].sampler, SequentialSampler)
    assert isinstance(loaders["test"].sampler, SequentialSampler)


def test_val_test_deterministic_order():
    _ensure_clean_dataset()

    def t_zero_seq(split):
        loader = build_dataloaders(DATASET_DIR, batch_size=64, num_workers=0)[split]
        return [m["t_zero"] for batch in loader for m in batch["metadata"]]

    assert t_zero_seq("val") == t_zero_seq("val")
    assert t_zero_seq("test") == t_zero_seq("test")


def test_seeded_train_shuffle_reproducible():
    _ensure_clean_dataset()
    loaders_a = build_dataloaders(DATASET_DIR, batch_size=64, num_workers=0, seed=7)
    loaders_b = build_dataloaders(DATASET_DIR, batch_size=64, num_workers=0, seed=7)
    a = [batch["metadata"] for batch in loaders_a["train"]]
    b = [batch["metadata"] for batch in loaders_b["train"]]
    assert a == b


def test_collate_metadata():
    _ensure_clean_dataset()
    ds = CycloneForecastingDataset(DATASET_DIR / "train.npz", DATASET_DIR / "train_metadata.csv")
    items = [ds[i] for i in range(3)]
    batch = collate_forecasting(items)
    assert batch["history"].shape == (3, 5, 7)
    assert batch["target"].shape == (3, 3, 3)
    assert len(batch["metadata"]) == 3
    assert all("cyclone_id" in m for m in batch["metadata"])


def test_item_fields():
    _ensure_clean_dataset()
    ds = CycloneForecastingDataset(DATASET_DIR / "train.npz", DATASET_DIR / "train_metadata.csv")
    item = ds[0]
    assert set(item.keys()) >= {"history", "target", "metadata"}
    assert item["history"].shape == (5, 7)
    assert item["target"].shape == (3, 3)
    assert item["history"].dtype == torch.float32