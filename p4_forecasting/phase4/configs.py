"""Phase-4 experiment configuration handling.

Six locked experiments:

    EXP001  Phase-3 reproduction with the Phase-4 16-feature input
            (improved_lstm 64/1/0.0 MSE)
    EXP002  Improved LSTM (96/2/0.1, MSE)
    EXP003  GRU (96/2/0.1, MSE)
    EXP004  Multi-task LSTM (96/2/0.1, MSE)
    EXP005  BEST architecture (from EXP002-004 validation) + Huber
    EXP006  BEST architecture + weighted multi-task loss

The "best architecture" for EXP005/006 is chosen with the VALIDATION criterion
(never test).  The runner rewrites EXP005/006 configs deterministically once
EXP001-004 validation metrics exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .common import FEATURE_NAMES, TARGET_NAMES

BASE = {
    "input_size": 16,
    "output_size": 9,
    "features": 16,
    "feature_order": FEATURE_NAMES,
    "target_order": TARGET_NAMES,
    "horizons_hours": [6, 12, 24],
    "seed": 42,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 15,
    "device": "cpu",
    "track_weight": 1.0,
    "intensity_weight": 1.0,
    "horizon_weights": [1.0, 1.0, 1.0],
    "huber_delta": 1.0,
    "loss_base": "mse",
}

FAMILY_DEFAULTS = {
    "improved_lstm": {"hidden_size": 96, "layers": 2, "dropout": 0.1},
    "gru": {"hidden_size": 96, "layers": 2, "dropout": 0.1},
    "multitask_lstm": {"hidden_size": 96, "layers": 2, "dropout": 0.1},
}

EXP_BASE = {
    "EXP001": {"model": "improved_lstm", "loss": "mse",
               "hidden_size": 64, "layers": 1, "dropout": 0.0,
               "note": "Phase-3 reproduction using the 16-feature contract"},
    "EXP002": {"model": "improved_lstm", "loss": "mse"},
    "EXP003": {"model": "gru", "loss": "mse"},
    "EXP004": {"model": "multitask_lstm", "loss": "mse"},
    "EXP005": {"model": None, "loss": "huber",
               "note": "best architecture (EXP002-004 validation) + Huber; "
                       "concretized on first run"},
    "EXP006": {"model": None, "loss": "weighted",
               "track_weight": 2.0, "intensity_weight": 1.0,
               "horizon_weights": [1.0, 1.0, 1.0],
               "note": "best architecture (EXP002-004 validation) + weighted "
                       "multi-task loss; concretized on first run"},
}


def make_config(exp_id: str) -> Dict[str, object]:
    base = dict(BASE)
    base["experiment_id"] = exp_id
    overrides = dict(EXP_BASE[exp_id])
    # Apply family defaults ONLY for experiments that do not pin their own
    # architecture hyperparameters explicitly (EXP001 pins 64/1/0.0 = the locked
    # Phase-3 reproduction; EXP002/003/004 take the family defaults; EXP005/006
    # are concretized later with the chosen best architecture).
    if overrides.get("model") in FAMILY_DEFAULTS and \
            not any(k in overrides for k in ("hidden_size", "layers", "dropout")):
        base.update(FAMILY_DEFAULTS[overrides["model"]])
    base.update(overrides)
    base.setdefault("hidden_size", 64)
    base.setdefault("layers", 1)
    base.setdefault("dropout", 0.0)
    return base


def write_config(cfg: Dict[str, object], path: Path) -> bool:
    path = Path(path)
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old == cfg:
                return False  # identical, do not rewrite
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def write_initial_configs(configs_dir: Path) -> List[str]:
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for eid in ("EXP001", "EXP002", "EXP003", "EXP004"):
        cfg = make_config(eid)
        if write_config(cfg, configs_dir / f"{eid}.json"):
            written.append(eid)
    return written


def select_best_architecture(
    val_results: Dict[str, Dict[str, object]],
) -> str:
    """Best architecture among EXP002/EXP003/EXP004 by validation criterion."""
    from .evaluation.selection import validation_primary_score

    candidates = {
        "EXP002": "improved_lstm",
        "EXP003": "gru",
        "EXP004": "multitask_lstm",
    }
    ranked = sorted(
        ((validation_primary_score(val_results[eid]), fam)
         for eid, fam in candidates.items() if eid in val_results),
        key=lambda t: t[0],
    )
    if not ranked:
        raise RuntimeError("no EXP002-004 validation results for architecture selection")
    return ranked[0][1]


def concretize_best_experiments(
    configs_dir: Path,
    best_arch: str,
) -> List[str]:
    """Rewrite EXP005/EXP006 with the chosen best architecture."""
    configs_dir = Path(configs_dir)
    written = []
    for eid in ("EXP005", "EXP006"):
        cfg = make_config(eid)
        cfg["model"] = best_arch
        cfg.update(FAMILY_DEFAULTS[best_arch])
        cfg["note"] = (f"architecture = {best_arch} (best of EXP002-004 on "
                       f"validation primary score)")
        if write_config(cfg, configs_dir / f"{eid}.json"):
            written.append(eid)
    return written