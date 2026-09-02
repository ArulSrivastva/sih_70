"""Deterministic Phase-4 experiment trainer.

CPU-only, seeded, de-terministic.  Uses TRAIN-ONLY statistics supplied by the
caller (results/normalization_stats.json).  The TEST split is never loaded
during training: only ``train.npz``/``val.npz`` are opened, and no test metric
is ever evaluated here or used for model selection.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..dataloader.forecasting_dataset import ForecastingDataset, collate_forecasting
from ..training.normalization import Normalizer

DEFAULT_SEED = 42
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 15


def set_all_seeds(seed: int, deterministic_cpu: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    seed: int = DEFAULT_SEED
    learning_rate: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    device: str = "cpu"
    force_retrain: bool = False


@dataclass
class TrainOutcome:
    reused: bool
    best_epoch: int
    best_val_loss: Optional[float]
    train_loss_at_best_epoch: Optional[float]
    epochs_run: int
    training_time_s: float
    parameter_count: int
    final_train_loss: Optional[float]
    final_val_loss: Optional[float]
    history: List[Dict[str, float]] = field(default_factory=list)
    device: str = "cpu"


def _load_split(dataset_dir: Path, split: str, normalizer: Normalizer):
    model_type = "raw"
    ds = ForecastingDataset(dataset_dir / f"{split}.npz", normalizer,
                            dataset_dir / f"{split}_metadata.csv")
    return ds


def _epoch_loss(model, loader, criterion, device) -> float:
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
    return total / n if n else float("nan")


def run_training(
    dataset_dir: Path,
    stats: Dict[str, object],
    checkpoint_path: Path,
    history_path: Path,
    cfg: TrainConfig,
    build_model: Callable[[], nn.Module],
    build_criterion: Callable[[], nn.Module],
) -> TrainOutcome:
    """Train (or restore) an experiment. Returns a TrainOutcome summary."""
    dataset_dir = Path(dataset_dir)
    for split in ("train", "val"):
        if not (dataset_dir / f"{split}.npz").exists():
            raise FileNotFoundError(f"missing dataset file: {dataset_dir / f'{split}.npz'}")

    set_all_seeds(cfg.seed)

    normalizer = Normalizer(stats)
    train_ds = _load_split(dataset_dir, "train", normalizer)
    val_ds = _load_split(dataset_dir, "val", normalizer)

    model = build_model()
    parameter_count = model.count_parameters()

    if checkpoint_path.exists() and not cfg.force_retrain:
        ckpt = torch.load(checkpoint_path, map_location=torch.device(cfg.device),
                          weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_epoch = int(ckpt["epoch"])
        best_val_loss = float(ckpt["val_loss"])
        epochs_so_far = int(ckpt.get("epochs_run", best_epoch))
        history: List[Dict[str, float]] = []
        if history_path.exists():
            try:
                saved = json.loads(history_path.read_text(encoding="utf-8"))
                history = [dict(e) for e in saved.get("history", [])]
            except Exception:
                history = [
                    {"epoch": e + 1, "loss": float("nan"), "val_loss": float("nan")}
                    for e in range(epochs_so_far)]
        print(f"[train] {checkpoint_path.stem}: checkpoint restored & reused "
              f"(best_epoch={best_epoch}, val_loss={best_val_loss:.6f}), no overwrite")
        return TrainOutcome(
            reused=True, best_epoch=best_epoch, best_val_loss=best_val_loss,
            train_loss_at_best_epoch=None, epochs_run=epochs_so_far,
            training_time_s=0.0, parameter_count=parameter_count,
            final_train_loss=None, final_val_loss=None, history=history,
            device=cfg.device)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_forecasting,
                              generator=g, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_forecasting, pin_memory=False)

    device = torch.device(cfg.device)
    model.to(device)
    criterion = build_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

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

        val_loss = _epoch_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "loss": float(train_loss), "val_loss": float(val_loss)})
        print(f"  epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

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
            "parameter_count": parameter_count,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, checkpoint_path)
        print(f"[train] best checkpoint saved -> {checkpoint_path} (epoch {best_epoch})")

    train_at_best = None
    if history and 0 <= best_epoch - 1 < len(history):
        train_at_best = float(history[best_epoch - 1]["loss"])

    json_out = {
        "seed": cfg.seed,
        "device": cfg.device,
        "best_epoch": best_epoch,
        "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
        "train_loss_at_best_epoch": train_at_best,
        "final_train_loss": float(history[-1]["loss"]) if history else None,
        "final_val_loss": float(history[-1]["val_loss"]) if history else None,
        "training_time_s": training_time,
        "epochs_run": len(history),
        "parameter_count": parameter_count,
        "history": history,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8")

    return TrainOutcome(
        reused=False, best_epoch=best_epoch,
        best_val_loss=None if best_val_loss == float("inf") else float(best_val_loss),
        train_loss_at_best_epoch=train_at_best, epochs_run=len(history),
        training_time_s=training_time, parameter_count=parameter_count,
        final_train_loss=json_out["final_train_loss"],
        final_val_loss=json_out["final_val_loss"], history=history, device=cfg.device)