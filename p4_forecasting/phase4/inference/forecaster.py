"""Phase-4 inference API for the 16-feature forecast contract.

Input  : numpy array (5, 16) or (1, 5, 16) in Phase-4 FEATURE ORDER (the 7 raw
         columns first, then the 9 engineered columns).
Output : dict {"forecast": [ {hours, latitude, longitude, wind_speed_kmh}, ... ]}
         for +6h / +12h / +24h.

Outputs are de-normalized with TRAIN-only statistics, and longitude is wrapped
consistently into [0, 360).  Wrong shapes, NaN or Inf are rejected, never
silently corrected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..common import FEATURE_NAMES, HORIZON_HOURS, TARGET_NAMES
from ..models import build_model
from ..training.normalization import Normalizer


def _wrap_lon(lon: float) -> float:
    return float(lon % 360.0)


class Phase4Forecaster:
    """Loads a champion checkpoint + config + stats and forecasts (5,16)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        stats_path: str | Path,
        device: str = "cpu",
    ) -> None:
        import torch
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.stats_path = Path(stats_path)
        for p in (self.checkpoint_path, self.config_path, self.stats_path):
            if not p.exists():
                raise FileNotFoundError(f"Phase4Forecaster missing file: {p}")
        with open(self.config_path, "r", encoding="utf-8") as fh:
            self.config: Dict[str, Any] = json.load(fh)
        self.normalizer = Normalizer.from_path(self.stats_path)
        self.model = build_model(None, self.config)
        ckpt = torch.load(self.checkpoint_path, map_location=torch.device("cpu"),
                          weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(torch.device(device))
        self.model.eval()

    def forecast(self, history: np.ndarray) -> Dict[str, Any]:
        """Forecast from a (5,16) or (1,5,16) engineered history block."""
        h = np.asarray(history, dtype=np.float32)
        if h.ndim == 2 and h.shape == (5, 16):
            h = h[None, ...]
        if h.ndim != 3 or h.shape[1:] != (5, 16):
            raise ValueError(
                f"history must be (5,16) or (1,5,16) with feature order "
                f"{FEATURE_NAMES}; got {h.shape}")
        if np.isnan(h).any() or np.isinf(h).any():
            raise ValueError("history contains NaN/Inf; refusing to forecast")

        import torch
        with torch.no_grad():
            xn = np.asarray(self.normalizer.normalize_X(h), dtype=np.float32)
            xt = torch.from_numpy(xn)
            pred = self.model(xt).squeeze(0).numpy()
            pred_denorm = np.asarray(self.normalizer.denormalize_Y(pred), dtype=np.float32)

        if np.isnan(pred_denorm).any() or np.isinf(pred_denorm).any():
            raise RuntimeError("forecast produced NaN/Inf; aborting")

        forecasts = []
        for i, hours in enumerate(HORIZON_HOURS):
            forecasts.append({
                "hours": int(hours),
                "latitude": float(pred_denorm[i, 0]),
                "longitude": _wrap_lon(float(pred_denorm[i, 1])),
                "wind_speed_kmh": float(pred_denorm[i, 2]),
            })
        return {"forecast": forecasts}