"""Phase-4 model evaluation on validation/test.

Predictions are denormalized with TRAIN-ONLY statistics; track error is measured
with the Phase-2 Haversine implementation (never raw Euclidean degrees).  The
TEST split is only ever evaluated through ``evaluate_once_for_champion`` and
never for model selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

from ..common import FEATURE_NAMES, TARGET_NAMES
from ..models import build_model
from ..training.normalization import Normalizer


def load_model_from_config(config_path: Path, checkpoint_path: Path, device: str = "cpu") -> nn.Module:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    model = build_model(None, cfg)
    ckpt = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_split(
    model: nn.Module,
    normalizer: Normalizer,
    dataset_dir: Path,
    split: str,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    npz = np.load(dataset_dir / f"{split}.npz", allow_pickle=True)
    X = np.asarray(npz["X"], dtype=np.float32)
    Y = np.asarray(npz["Y"], dtype=np.float32)
    if X.shape[-1] != 16:
        raise ValueError(f"{split}: X must have 16 features; got {X.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"{split}: X/Y row mismatch")

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            Xb = X[i:i + batch_size]
            Xn = np.asarray(normalizer.normalize_X(Xb), dtype=np.float32)
            Xt = torch.from_numpy(Xn)
            preds.append(model(Xt).numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_pred = np.asarray(normalizer.denormalize_Y(y_pred), dtype=np.float32)
    return {"y_true": np.asarray(Y, np.float32), "y_pred": y_pred}


def evaluate_split(
    model: nn.Module,
    normalizer: Normalizer,
    dataset_dir: Path,
    split: str,
    batch_size: int = 256,
) -> Dict[str, object]:
    """Return the Phase-2 metric summary for a split plus counts."""
    from phase2.evaluation.metrics import summarize_split

    pred = predict_split(model, normalizer, dataset_dir, split, batch_size=batch_size)
    metrics = summarize_split(pred["y_true"], pred["y_pred"])
    return {
        "split": split,
        "samples": int(pred["y_true"].shape[0]),
        "metrics": metrics,
    }


def evaluate_validation(model, normalizer, dataset_dir, exp_dir: Path) -> Dict[str, object]:
    result = evaluate_split(model, normalizer, dataset_dir, "val")
    return result


def write_json(obj: Dict[str, object], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")