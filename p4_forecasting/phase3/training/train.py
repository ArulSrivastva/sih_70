"""Training pipeline for P4 Phase 3.

Deterministic training of the lightweight LSTM:
  1. load Phase-2 CLEAN chronological dataset (train + val only)
  2. compute training-only normalization statistics
  3. normalize train/val with train statistics
  4. create the LSTM
  5. train with MSELoss (normalized targets), Adam lr=1e-3, batch_size=64
  6. evaluate validation loss after each epoch
  7. early stop (patience=12, monitor validation loss)
  8. save best-validation checkpoint under phase3/checkpoints/
  9. save training history under phase3/results/training_history.json

The model is NEVER selected using test performance, and the test split is not
loaded here at all (no test-stat leakage by construction).

Seeds are fixed for reproducibility: python/numpy/torch, deterministic CPU.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..model.cyclone_lstm import CycloneLSTM
from .dataset import NormalizedCycloneDataset, collate_normalized
from .normalization import FEATURES, TARGETS, Normalizer, compute_normalization_stats

DEFAULT_SEED = 42
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 12
FEATURE_ORDER = FEATURES
TARGET_ORDER = TARGETS


def set_all_seeds(seed: int, deterministic_cpu: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: Optional[float]
    history: List[Dict[str, float]] = field(default_factory=list)
    epochs_run: int = 0
    training_time_s: float = 0.0
    parameter_count: int = 0
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    device: str = "cpu"


@dataclass
class TrainConfig:
    seed: int = DEFAULT_SEED
    learning_rate: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    hidden_size: int = 64
    num_layers: int = 1
    device: str = "cpu"  # deterministic by design
    force_retrain: bool = False


def _load_raw(split: str, dataset_dir: Path) -> tuple:
    npz = np.load(dataset_dir / f"{split}.npz", allow_pickle=True)
    meta = None
    meta_path = dataset_dir / f"{split}_metadata.csv"
    if meta_path.exists():
        import pandas as pd
        meta = pd.read_csv(meta_path)
    return np.asarray(npz["X"], np.float32), np.asarray(npz["Y"], np.float32), meta


def _epoch_avg_loss(model, loader, criterion, device, train: bool = False) -> float:
    """Average loss over a loader in no-grad mode (used for validation scoring)."""
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["history"].to(device)
            y = batch["target"].to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total += float(loss.item()) * x.shape[0]
            n += x.shape[0]
    if n == 0:
        return float("nan")
    return total / n


def run_training(
    dataset_dir: Path,
    stats_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    history_path: Path,
    cfg: TrainConfig,
) -> TrainResult:
    """Run or restore training. Returns the training result summary."""
    dataset_dir = Path(dataset_dir)
    for split in ("train", "val"):
        if not (dataset_dir / f"{split}.npz").exists():
            raise FileNotFoundError(f"Missing dataset file: {dataset_dir / f'{split}.npz'}")

    set_all_seeds(cfg.seed)

    X_train, Y_train, meta_train = _load_raw("train", dataset_dir)
    X_val, Y_val, meta_val = _load_raw("val", dataset_dir)

    stats = compute_normalization_stats(X_train, Y_train)
    stats["training_seed"] = cfg.seed
    normalizer = Normalizer(stats)
    normalizer.save(stats_path)

    model = CycloneLSTM(hidden_size=cfg.hidden_size, num_layers=cfg.num_layers)
    param_count = model.count_parameters()

    model_config = {
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "output_size": model.output_size,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "feature_order": FEATURE_ORDER,
        "target_order": TARGET_ORDER,
        "horizons_hours": [6, 12, 24],
        "parameters": param_count,
        "architecture": "LSTM(7->64) -> Linear(64->9) -> reshape(3,3)",
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(model_config, indent=2, ensure_ascii=False), encoding="utf-8")

    if checkpoint_path.exists() and not cfg.force_retrain:
        ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"),
                          weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_epoch = int(ckpt["epoch"])
        best_val_loss = float(ckpt["val_loss"])
        epochs_so_far = int(ckpt.get("epochs_run", best_epoch))
        print(f"[train] checkpoint exists -> restored (best epoch {best_epoch}, "
              f"val_loss={best_val_loss:.6f}); reusing, no overwrite.")
        return TrainResult(
            best_epoch=best_epoch, best_val_loss=best_val_loss,
            epochs_run=epochs_so_far, parameter_count=param_count,
            device=cfg.device, history=[
                {"epoch": e + 1, "loss": float("nan"), "val_loss": float("nan")}
                for e in range(epochs_so_far)],
        )

    train_ds = NormalizedCycloneDataset(dataset_dir / "train.npz", normalizer,
                                        dataset_dir / "train_metadata.csv")
    val_ds = NormalizedCycloneDataset(dataset_dir / "val.npz", normalizer,
                                      dataset_dir / "val_metadata.csv")

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_normalized, generator=g, pin_memory=False)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_normalized, pin_memory=False)

    device = torch.device(cfg.device)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Explicit per-batch optimizer loop (deterministic, single backward/step per batch).
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_hits = 0
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in train_loader:
            x = batch["history"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        train_loss = epoch_loss / n

        model.eval()
        val_loss = _epoch_avg_loss(model, val_loader, criterion, device, train=False)

        history.append({"epoch": epoch, "loss": float(train_loss), "val_loss": float(val_loss)})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_hits = 0
        else:
            patience_hits += 1
            if patience_hits >= cfg.patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(no improvement for {cfg.patience} epochs)")
                break

    training_time = time.time() - t0

    if best_state is not None:
        ckpt = {
            "state_dict": best_state,
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "epochs_run": len(history),
            "seed": cfg.seed,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, checkpoint_path)
        print(f"[train] best checkpoint saved -> {checkpoint_path} (epoch {best_epoch})")

    result = TrainResult(
        best_epoch=best_epoch,
        best_val_loss=None if best_val_loss == float("inf") else float(best_val_loss),
        history=history,
        epochs_run=len(history),
        training_time_s=training_time,
        parameter_count=param_count,
        final_train_loss=float(history[-1]["loss"]) if history else None,
        final_val_loss=float(history[-1]["val_loss"]) if history else None,
        device=cfg.device,
    )

    json_out = {
        "seed": cfg.seed,
        "device": cfg.device,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "final_train_loss": result.final_train_loss,
        "final_val_loss": result.final_val_loss,
        "train_loss_at_best_epoch": (
            float(history[result.best_epoch - 1]["loss"]) if history and result.best_epoch - 1 < len(history) else None),
        "training_time_s": result.training_time_s,
        "epochs_run": result.epochs_run,
        "parameter_count": result.parameter_count,
        "history": history,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8")
    return result