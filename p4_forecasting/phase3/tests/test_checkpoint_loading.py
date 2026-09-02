""". Checkpoint loading + reuse tests for P4 Phase 3.

Includes a self-contained save/load roundtrip and (when trained) checks that the
saved best checkpoint actually loads and reproduces the training architecture.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ..model.cyclone_lstm import CycloneLSTM
from ..training.dataset import NormalizedCycloneDataset, collate_normalized
from ..training.normalization import Normalizer, compute_normalization_stats

P3_DIR = Path(__file__).resolve().parents[1]
CKPT = P3_DIR / "checkpoints" / "best_lstm.pt"
CONFIG = P3_DIR / "checkpoints" / "model_config.json"
STATS = P3_DIR / "results" / "normalization_stats.json"


def test_state_dict_roundtrip():
    torch = pytest.importorskip("torch")
    model = CycloneLSTM()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.pt"
        torch.save({"state_dict": model.state_dict(), "epoch": 1, "val_loss": 0.5}, p)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        restored = CycloneLSTM()
        restored.load_state_dict(ckpt["state_dict"])
        x = torch.randn(2, 5, 7)
        assert torch.allclose(model(x), restored(x), atol=1e-6)


def test_dataset_normalized_shapes():
    rng = np.random.default_rng(3)
    X = rng.normal(loc=0, scale=1, size=(8, 5, 7)).astype(np.float32)
    Y = rng.normal(loc=0, scale=1, size=(8, 3, 3)).astype(np.float32)
    n = Normalizer(compute_normalization_stats(X, Y))
    with tempfile.TemporaryDirectory() as tmp:
        import numpy as _np
        p = Path(tmp) / "t.npz"
        _np.savez(p, X=X, Y=Y)
        ds = NormalizedCycloneDataset(p, normalizer=n)
        item = ds[0]
        assert item["history"].shape == (5, 7)
        assert item["target"].shape == (3, 3)
        b = collate_normalized([ds[0], ds[1]])
        assert b["history"].shape == (2, 5, 7)
        assert b["target"].shape == (2, 3, 3)


@pytest.mark.skipif(not (CKPT.exists() and CONFIG.exists()),
                    reason="Phase-3 checkpoint not trained yet")
def test_best_checkpoint_loads():
    import torch
    import json
    with open(CONFIG, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    model = CycloneLSTM(input_size=cfg["input_size"], hidden_size=cfg["hidden_size"],
                        num_layers=cfg["num_layers"], output_size=cfg["output_size"])
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    assert isinstance(ckpt["epoch"], int)
    assert ckpt["val_loss"] >= 0.0
    out = model(torch.randn(1, 5, 7))
    assert out.shape == (1, 3, 3)