"""One-experiment runner for Phase-4 (used by run_phase4.py STEP 8).

Wraps training + validation evaluation + artifact writes (config.json,
training_history.json, validation_results.json, checkpoint, metrics.json,
source_hashes.json).  A failed experiment is captured in its summary with
status=FAILED and is never hidden.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Callable, Dict, List

import torch
from torch import nn

from .common import sha256
from .configs import make_config, write_config
from .evaluation.evaluate import evaluate_split, write_json
from .evaluation.selection import tiebreak_values, validation_primary_score
from .losses import build_criterion
from .models import build_model
from .training.normalization import Normalizer
from .training.train import TrainConfig, run_training

CHECKPOINT_NAME = "checkpoint.pt"


def _build_model(config: Dict[str, object]) -> nn.Module:
    return build_model(None, config)


def _build_criterion(config: Dict[str, object]) -> Callable[[], nn.Module]:
    def factory() -> nn.Module:
        return build_criterion(config)
    return factory


def run_one_experiment(
    exp_id: str,
    configs_dir: Path,
    dataset_dir: Path,
    stats_path: Path,
    results_dir: Path,
    immutable_sources: Dict[str, str],
    force_retrain: bool,
) -> Dict[str, object]:
    out_dir = results_dir / "experiments" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / CHECKPOINT_NAME
    history_path = out_dir / "training_history.json"
    val_path = out_dir / "validation_results.json"
    metrics_path = out_dir / "metrics.json"
    source_hashes_path = out_dir / "source_hashes.json"

    cfg = make_config(exp_id) if not (configs_dir / f"{exp_id}.json").exists() \
        else json.loads((configs_dir / f"{exp_id}.json").read_text(encoding="utf-8"))
    cfg["experiment_id"] = exp_id
    copied = write_config(cfg, out_dir / "config.json")
    if copied:
        print(f"[experiment] {exp_id}: config copied to experiment dir")

    stats = Normalizer.from_path(stats_path)

    train_cfg = TrainConfig(
        seed=int(cfg.get("seed", 42)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        batch_size=int(cfg.get("batch_size", 64)),
        max_epochs=int(cfg.get("max_epochs", 100)),
        patience=int(cfg.get("patience", 15)),
        device=str(cfg.get("device", "cpu")),
        force_retrain=bool(force_retrain),
    )

    try:
        outcome = run_training(
            dataset_dir=dataset_dir,
            stats=stats.stats,
            checkpoint_path=checkpoint,
            history_path=history_path,
            cfg=train_cfg,
            build_model=lambda: _build_model(cfg),
            build_criterion=_build_criterion(cfg),
        )
    except Exception:
        tb = traceback.format_exc()
        print(f"[experiment] {exp_id}: TRAINING FAILED")
        print(tb)
        write_json({"error": tb, "where": "training"}, out_dir / "error.json")
        return _failed_summary(exp_id, cfg, "training", tb)

    # Validation-only evaluation.
    try:
        model = _load_for_eval(cfg, checkpoint)
        normalizer = Normalizer.from_path(stats_path)
        val_result = evaluate_split(model, normalizer, dataset_dir, "val")
    except Exception:
        tb = traceback.format_exc()
        print(f"[experiment] {exp_id}: VALIDATION EVALUATION FAILED")
        print(tb)
        write_json({"error": tb, "where": "validation"}, out_dir / "error.json")
        return _failed_summary(exp_id, cfg, "validation", tb)

    metrics_events = val_result["metrics"]
    write_json(val_result, val_path)
    val_metrics = val_result["metrics"]
    metrics = {
        "experiment_id": exp_id,
        "split": "val",
        "samples": val_result["samples"],
        "best_epoch": outcome.best_epoch,
        "best_val_loss": outcome.best_val_loss,
        "train_loss_at_best_epoch": outcome.train_loss_at_best_epoch,
        "epochs_run": outcome.epochs_run,
        "training_time_s": outcome.training_time_s,
        "parameter_count": outcome.parameter_count,
        "checkpoint_reused": outcome.reused,
        "validation_primary_score": validation_primary_score(val_metrics),
        "tiebreak": {"mean_wind_mae": tiebreak_values(val_metrics)[0],
                     "mean_wind_rmse": tiebreak_values(val_metrics)[1]},
        "per_horizon_track_mean_km": {
            h: val_metrics[h]["track_error_km_mean"] for h in ("6h", "12h", "24h")},
        "per_horizon_wind_mae": {
            h: val_metrics[h]["wind_mae"] for h in ("6h", "12h", "24h")},
    }
    write_json(metrics, metrics_path)

    src_hashes = {
        "experiment_id": exp_id,
        "consumed": {
            "dataset": {
                "train.npz": sha256(dataset_dir / "train.npz"),
                "val.npz": sha256(dataset_dir / "val.npz"),
            },
            "normalization_stats.json": sha256(stats_path),
            "config.json": sha256(out_dir / "config.json"),
        },
        "immutable_sources": immutable_sources,
    }
    write_json(src_hashes, source_hashes_path)

    print(f"[experiment] {exp_id}: validation primary={metrics['validation_primary_score']:.3f} "
          f"status=PASS (reused={outcome.reused})")
    return {
        "experiment_id": exp_id,
        "status": "PASS",
        "config": cfg,
        "best_epoch": outcome.best_epoch,
        "best_val_loss": outcome.best_val_loss,
        "epochs_run": outcome.epochs_run,
        "training_time_s": outcome.training_time_s,
        "parameter_count": outcome.parameter_count,
        "checkpoint_reused": outcome.reused,
        "validation_metrics": val_metrics,
        "validation_primary_score": metrics["validation_primary_score"],
        "dir": str(out_dir),
    }


def _load_for_eval(config: Dict[str, object], checkpoint: Path):
    from .evaluation.evaluate import load_model_from_config
    return load_model_from_config(checkpoint.parent / "config.json", checkpoint)


def _failed_summary(exp_id: str, cfg: Dict[str, object], stage: str, tb: str) -> Dict[str, object]:
    return {
        "experiment_id": exp_id,
        "status": "FAILED",
        "stage": stage,
        "error": tb,
        "config": cfg,
        "best_epoch": None,
        "best_val_loss": None,
        "validation_metrics": {},
    }