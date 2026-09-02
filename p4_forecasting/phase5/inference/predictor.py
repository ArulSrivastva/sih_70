"""Phase-5 model predictor (read-only wrapper around the Phase-4 champion).

Loads the audited EXP005 checkpoint without retraining, applies train-only
normalization, runs the GRU and de-normalizes.  Executes deterministically on
CPU.  Output contract enforcement (lon -> [0,360), lat within [-90,90] else
hard error, wind >= 0) is applied after de-normalization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config import HORIZON_HOURS, N_TARGETS


def configure_deterministic_cpu() -> None:
    """Single-threaded CPU determinism for repeatable inference."""
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.enabled = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _wrap_lon(lon: float) -> float:
    return float(lon % 360.0)


class CyclonePredictor:
    """Champion GRU + train-only normalization, exposed as a plain class."""

    def __init__(self, checkpoint_path: str | Path,
                 config_path: str | Path,
                 stats_path: str | Path,
                 device: str = "cpu") -> None:
        import torch
        configure_deterministic_cpu()
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.stats_path = Path(stats_path)
        for p in (self.checkpoint_path, self.config_path, self.stats_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"CyclonePredictor missing required file: {p}")

        with open(self.config_path, "r", encoding="utf-8") as fh:
            self.config: Dict[str, Any] = json.load(fh)
        if self.config.get("model") != "gru":
            raise ValueError(
                f"champion config model is '{self.config.get('model')}', "
                f"expected 'gru'")

        from phase4.models.gru import GRUCyclone
        from phase4.training.normalization import Normalizer

        self.normalizer = Normalizer.from_path(stats_path)
        self.model = GRUCyclone(
            input_size=int(self.config.get("input_size", 16)),
            hidden_size=int(self.config.get("hidden_size", 64)),
            num_layers=int(self.config.get("layers", 1)),
            output_size=int(self.config.get("output_size", 9)),
            dropout=float(self.config.get("dropout", 0.0)),
        )
        ckpt = torch.load(self.checkpoint_path, map_location=torch.device("cpu"),
                          weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(torch.device(device))
        self.model.eval()
        self._ckpt = ckpt
        self.device = device

        self.param_count = sum(v.numel() for v in self.model.parameters())
        self.horizon_hours = [int(h) for h in self.config["horizons_hours"]]
        if self.horizon_hours != HORIZON_HOURS:
            raise ValueError(
                f"config horizons {self.horizon_hours} != {HORIZON_HOURS}")

    # -- model introspection -------------------------------------------------
    @property
    def feature_count(self) -> int:
        return int(self.config.get("input_size", 16))

    @property
    def history_steps(self) -> int:
        return 5

    def model_info(self, experiment_id: str) -> Dict[str, str]:
        family = {"gru": "GRU", "improved_lstm": "ImprovedLSTM",
                  "multitask_lstm": "MultiTaskLSTM"}.get(
            str(self.config.get("model")), str(self.config.get("model")))
        return {
            "experiment_id": experiment_id,
            "family": family,
            "loss": str(self.config.get("loss", "")).title(),
        }

    # -- prediction ----------------------------------------------------------
    def predict_features(self, features16: np.ndarray) -> np.ndarray:
        """(5,16) engineered features -> (3,3) de-normalised physical targets.

        Rows = +6h/+12h/+24h; cols = [lat, lon, wind_speed_kmh].
        """
        import torch
        x = np.asarray(features16, dtype=np.float32)
        if x.shape != (5, 16):
            raise ValueError(f"expected (5,16) features; got {x.shape}")
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("features contain NaN/Inf; refusing to forecast")

        xn = np.asarray(self.normalizer.normalize_X(x[None, ...]), np.float32)
        xt = torch.from_numpy(xn)
        with torch.no_grad():
            pred = np.asarray(
                self.model(xt).squeeze(0).numpy(), dtype=np.float32)
        phys = np.asarray(self.normalizer.denormalize_Y(pred), np.float32)

        if np.isnan(phys).any() or np.isinf(phys).any():
            raise RuntimeError("forecast produced NaN/Inf; aborting")

        lat, lon, wind = phys[:, 0], phys[:, 1], phys[:, 2]
        if float(np.abs(lat).max()) > 90.0:
            raise RuntimeError(
                "forecast latitude outside [-90, 90]; refusing to emit "
                "impossible coordinates")
        phys[:, 1] = np.array([_wrap_lon(v) for v in lon], dtype=np.float32)
        phys[:, 2] = np.maximum(phys[:, 2], 0.0)          # contract: wind >= 0
        return phys

    def predict_from_history(self, history7: np.ndarray) -> np.ndarray:
        from .preprocessing import engineer_history
        feats = engineer_history(np.asarray(history7, np.float32))
        return self.predict_features(feats)

    # -- determinism helper --------------------------------------------------
    def forecast_twice(self, features16: np.ndarray):
        a = self.predict_features(features16)
        b = self.predict_features(features16)
        exact = bool(np.array_equal(a, b))
        max_abs = float(np.max(np.abs(a - b))) if not exact else 0.0
        return {"exact_equal": exact, "max_abs_diff": max_abs,
                "tolerance": 1e-6, "pass": exact or max_abs <= 1e-6}


def load_predictor(paths, device: str = "cpu") -> CyclonePredictor:
    return CyclonePredictor(paths.champion_checkpoint,
                            paths.champion_config,
                            paths.normalization_stats,
                            device=device)