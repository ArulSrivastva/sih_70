"""Unit tests for the three Phase-4 model families."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phase4.models import MODEL_FAMILIES, build_model
from phase4.models.gru import GRUCyclone
from phase4.models.improved_lstm import ImprovedLSTM
from phase4.models.multitask_lstm import MultiTaskLSTM

FAMILIES = ["improved_lstm", "gru", "multitask_lstm"]


def _cfg(model, **over):
    cfg = {
        "model": model, "input_size": 16, "output_size": 9,
        "hidden_size": 24, "layers": 2, "dropout": 0.0,
        "num_layers": 2,
    }
    cfg.update(over)
    return cfg


def _input(batch=4, features=16, seq=5):
    return torch.randn(batch, seq, features)


@pytest.mark.parametrize("family", FAMILIES)
def test_forward_shapes(family):
    model = build_model(None, _cfg(family))
    out = model(_input())
    assert out.shape == (4, 3, 3)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_forward_numpy(family):
    model = build_model(None, _cfg(family))
    tensor, arr = model.forward_numpy(np.random.randn(2, 5, 16).astype(np.float32))
    assert tensor.shape == (2, 3, 3)
    assert arr.shape == (2, 3, 3)
    assert np.isfinite(arr).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_rejects_wrong_input_shape(family):
    model = build_model(None, _cfg(family))
    with pytest.raises(ValueError):
        model(torch.randn(4, 5, 8))       # wrong features
    with pytest.raises(ValueError):
        model(torch.randn(4, 4, 16))      # wrong steps
    with pytest.raises(ValueError):
        model(torch.randn(4, 5, 16, 1))   # too many dims


@pytest.mark.parametrize("family", FAMILIES)
def test_parameter_count_positive(family):
    model = build_model(None, _cfg(family))
    assert model.count_parameters() > 0


def test_multitask_composition():
    model = build_model(None, _cfg("multitask_lstm", hidden_size=16))
    out = model(_input(8))
    # last channel is wind (intensity head), first two lat/lon (track head)
    assert out.ndim == 3 and out.shape[1:] == (3, 3)
    track = out[..., :2]
    intensity = out[..., 2:3]
    assert torch.allclose(torch.cat([track, intensity], dim=-1), out)


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        build_model(None, _cfg("not_a_model"))


def test_deterministic_forward_without_dropout():
    torch.manual_seed(11)
    a = build_model(None, _cfg("improved_lstm", dropout=0.0))
    torch.manual_seed(11)
    b = build_model(None, _cfg("improved_lstm", dropout=0.0))
    x = torch.randn(4, 5, 16)
    a.eval()
    b.eval()
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_family_registry_complete():
    assert set(MODEL_FAMILIES) == set(FAMILIES)