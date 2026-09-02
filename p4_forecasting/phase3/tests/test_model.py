""". Self-contained tests for the P4 Phase 3 LSTM model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ..model.cyclone_lstm import CycloneLSTM, INPUT_SIZE, OUTPUT_SIZE, N_HORIZONS, N_TARGETS


def _model() -> CycloneLSTM:
    return CycloneLSTM()


def test_input_shape_single():
    m = _model().eval()
    x = torch.randn(1, 5, 7)
    out = m(x)
    assert out.shape == (1, 3, 3)


def test_input_shape_batch():
    m = _model().eval()
    x = torch.randn(4, 5, 7)
    out = m(x)
    assert out.shape == (4, 3, 3)


def test_output_no_nan():
    m = _model().eval()
    out = m(torch.randn(8, 5, 7))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_wrong_feature_count_fails():
    m = _model()
    with pytest.raises(ValueError):
        m(torch.randn(5, 6))


def test_wrong_history_steps_fails():
    m = _model()
    with pytest.raises(ValueError):
        m(torch.randn(4, 4, 7))


def test_config_defaults_consistent():
    m = _model()
    assert m.input_size == INPUT_SIZE == 7
    assert m.output_size == OUTPUT_SIZE == 9
    assert N_HORIZONS == 3 and N_TARGETS == 3


def test_parameter_count():
    m = _model()
    params = m.count_parameters()
    # LSTM(7->64): W_ih 256*7 + W_hh 256*64 + b_ih 256 + b_hh 256 = 18688
    # Linear(64->9): 64*9 + 9 = 585  => total = 19273
    assert params == 19273, params


def test_deterministic_forward():
    torch.manual_seed(0)
    m = _model().eval()
    x = torch.randn(3, 5, 7)
    o1 = m(x)
    torch.manual_seed(0)
    m2 = _model().eval()
    o2 = m2(x)
    assert torch.allclose(o1, o2, atol=1e-6)