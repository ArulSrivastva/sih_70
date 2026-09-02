"""Unit tests for the Phase-4 DataLoader / Dataset contract."""

from __future__ import annotations

import numpy as np
import torch

import pytest

from phase4.dataloader.forecasting_dataset import (
    ForecastingDataset,
    collate_forecasting,
)
from phase4.training.normalization import Normalizer
from tests.conftest import make_feature_npz, make_normalization_stats


def test_dataset_shapes_and_metadata(scratch):
    ds_dir = make_feature_npz(scratch / "ds", n_train=20, n_val=5)
    stats = make_normalization_stats()
    norm = Normalizer(stats)
    ds = ForecastingDataset(ds_dir / "train.npz", norm, ds_dir / "train_metadata.csv")
    assert len(ds) == 20
    item = ds[3]
    assert tuple(item["history"].shape) == (5, 16)
    assert tuple(item["target"].shape) == (3, 3)
    assert item["metadata"]["cyclone_id"] != ""  # metadata passthrough
    # normalized sample history stays centered near zero (loose tolerance)
    assert abs(float(item["history"].mean())) < 2.0


def test_dataset_rejects_bad_contract(scratch):
    ds_dir = make_feature_npz(scratch / "ds2", n_train=10, n_val=2)
    # corrupt the feature dimension to 15
    z = np.load(ds_dir / "train.npz", allow_pickle=True)
    bad = ds_dir / "bad.npz"
    np.savez_compressed(bad, X=np.zeros((4, 5, 15), dtype=np.float32),
                        Y=np.zeros((4, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        ForecastingDataset(bad)


def test_collate_batches(scratch):
    ds_dir = make_feature_npz(scratch / "ds3", n_train=9, n_val=3)
    ds = ForecastingDataset(ds_dir / "train.npz")
    batch = collate_forecasting([ds[i] for i in range(6)])
    assert tuple(batch["history"].shape) == (6, 5, 16)
    assert tuple(batch["target"].shape) == (6, 3, 3)
    assert isinstance(batch["history"], torch.Tensor)


def test_no_target_normalization_when_disabled(scratch):
    ds_dir = make_feature_npz(scratch / "ds4", n_train=6, n_val=2)
    stats = make_normalization_stats()
    ds = ForecastingDataset(ds_dir / "train.npz", Normalizer(stats),
                            normalize_targets=False)
    raw = np.load(ds_dir / "train.npz")["Y"]
    assert np.allclose(ds[0]["target"].numpy(), raw[0].astype(np.float32), atol=1e-5)