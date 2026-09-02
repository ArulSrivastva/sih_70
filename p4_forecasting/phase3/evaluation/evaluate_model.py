"""Evaluation of the best LSTM checkpoint on validation and test (P4 Phase 3).

The TEST set is evaluated exactly once for official final metrics, after model
selection. All predictions are denormalized with TRAINING-only statistics, then
scored with the Phase-2 metric implementation (Haversine + wind MAE/RMSE).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ..model.cyclone_lstm import CycloneLSTM
from ..training.normalization import Normalizer

try:  # reuse Phase-2 metric implementation (do not duplicate)
    from phase2.evaluation.metrics import summarize_split  # noqa: F401
except ImportError:
    raise ImportError(
        "`phase2` is not importable: add p4_forecasting/ to sys.path (run_phase3.py "
        "does this automatically) before importing phase3.evaluation.")


def build_model_from_config(config_path: Path, checkpoint_path: Path) -> CycloneLSTM:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    model = CycloneLSTM(
        input_size=int(cfg.get("input_size", 7)),
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 1)),
        output_size=int(cfg.get("output_size", 9)),
    )
    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"),
                      weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_split(
    model: CycloneLSTM,
    normalizer: Normalizer,
    dataset_dir: Path,
    split: str,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Return denormalized (y_true, y_pred) arrays of shape (N,3,3) for a split."""
    npz = np.load(dataset_dir / f"{split}.npz", allow_pickle=True)
    X = np.asarray(npz["X"], dtype=np.float32)
    Y = np.asarray(npz["Y"], dtype=np.float32)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"{split}: X/Y row mismatch")

    with torch.no_grad():
        model.eval()
        Xn = np.asarray(normalizer.normalize_X(X), dtype=np.float32)
        Xt = torch.from_numpy(Xn)
        pred = model(Xt)
        y_pred_norm = pred.numpy()
    y_pred = np.asarray(normalizer.denormalize_Y(y_pred_norm), dtype=np.float32)
    return {"y_true": np.asarray(Y, np.float32), "y_pred": y_pred}


def evaluate_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    stats_path: Path,
    dataset_dir: Path,
    device: str = "cpu",
) -> Dict[str, object]:
    """Evaluate the best checkpoint on validation and test. Returns full metrics."""
    model = build_model_from_config(config_path, checkpoint_path)
    normalizer = Normalizer.from_path(stats_path)

    results: Dict[str, object] = {"primary_split": "test", "val": {}, "test": {}}
    for split in ("val", "test"):
        pred = predict_split(model, normalizer, dataset_dir, split, device)
        results[split] = summarize_split(pred["y_true"], pred["y_pred"])

    return results


def save_evaluation_results(results: Dict[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")