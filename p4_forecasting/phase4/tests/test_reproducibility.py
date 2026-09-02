"""Reproducibility tests: seeded runs must be bit-identical checkpoint-agnostic."""

from __future__ import annotations

import numpy as np
import torch

from phase4.training.train import set_all_seeds
from tests.conftest import make_feature_npz, make_normalization_stats


def test_set_all_seeds_is_repeatable():
    set_all_seeds(1234)
    a = torch.randn(50)
    set_all_seeds(1234)
    b = torch.randn(50)
    assert torch.equal(a, b)


def test_dataengineer_reproducible(synthetic_raw_history):
    """Feature engineering is a pure function: same input -> identical output."""
    from phase4.features.feature_engineering import engineer_features

    X = synthetic_raw_history
    f1 = engineer_features(X)
    f2 = engineer_features(X.copy())
    assert np.array_equal(f1, f2)


def test_normalization_stats_reproducible():
    rng = np.random.RandomState(99)
    X = rng.uniform(size=(60, 5, 16)).astype(np.float32)
    Y = rng.uniform(size=(60, 3, 3)).astype(np.float32)
    from phase4.training.normalization import compute_normalization_stats

    s1 = compute_normalization_stats(X, Y)
    s2 = compute_normalization_stats(X.copy(), Y.copy())
    assert s1["feature_mean"] == s2["feature_mean"]
    assert s1["feature_std"] == s2["feature_std"]
    assert s1["target_mean"] == s2["target_mean"]


def test_model_weights_reproducible_under_same_seed():
    from phase4.models import ImprovedLSTM

    torch.manual_seed(5)
    m1 = ImprovedLSTM(input_size=16, hidden_size=16, num_layers=2, output_size=9)
    state1 = {k: v.clone() for k, v in m1.state_dict().items()}
    torch.manual_seed(5)
    m2 = ImprovedLSTM(input_size=16, hidden_size=16, num_layers=2, output_size=9)
    for k in state1:
        assert torch.equal(state1[k], m2.state_dict()[k])


def test_dataloader_batches_reproducible(scratch):
    from phase4.dataloader.forecasting_dataset import (
        ForecastingDataset,
        collate_forecasting,
    )
    from torch.utils.data import DataLoader

    from phase4.training.train import set_all_seeds

    ds = make_feature_npz(scratch / "rep", n_train=16, n_val=4)

    def first_batch():
        set_all_seeds(77)
        g = torch.Generator()
        g.manual_seed(77)
        loader = DataLoader(ForecastingDataset(ds / "train.npz"), batch_size=8,
                            shuffle=True, generator=g, collate_fn=collate_forecasting)
        return loader  # will be iterated below

    batches = []
    for _ in range(2):
        b = next(iter(first_batch()))
        batches.append(b["history"].numpy())
    assert np.array_equal(batches[0], batches[1])