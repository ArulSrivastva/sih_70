"""Inference API for P4 Phase 3.

CycloneForecaster loads best_lstm.pt + normalization_stats.json + model_config.json
and produces de-normalized forecasts in P4 integration-contract format.

Input  : numpy array, shape (5, 7) in FEATURE ORDER
         [lat, lon, wind_speed, pressure, sst, wind_u, wind_v] (SST in degrees C)
Output : dict
         {"forecast": [
             {"hours": 6,  "latitude": ..., "longitude": ..., "wind_speed_kmh": ...},
             {"hours": 12, "latitude": ..., "longitude": ..., "wind_speed_kmh": ...},
             {"hours": 24, "latitude": ..., "longitude": ..., "wind_speed_kmh": ...}]}

Longitude is wrapped into the P1 [0, 360) convention. No future/target data is
used: the forecast depends only on the 5-step history input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..model.cyclone_lstm import CycloneLSTM
from ..training.normalization import Normalizer

FEATURES = ["lat", "lon", "wind_speed", "pressure", "sst", "wind_u", "wind_v"]
HORIZONS_HOURS = [6, 12, 24]


def _wrap_lon(lon: float) -> float:
    return float(lon % 360.0)


class CycloneForecaster:
    """Loads checkpoint + stats + config and forecasts from a (5,7) history."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        stats_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.stats_path = Path(stats_path)
        for p in (self.checkpoint_path, self.config_path, self.stats_path):
            if not p.exists():
                raise FileNotFoundError(f"CycloneForecaster missing required file: {p}")

        with open(self.config_path, "r", encoding="utf-8") as fh:
            self.config: Dict[str, Any] = json.load(fh)
        self.device = device
        self.normalizer = Normalizer.from_path(self.stats_path)
        self.model = CycloneLSTM(
            input_size=int(self.config.get("input_size", 7)),
            hidden_size=int(self.config.get("hidden_size", 64)),
            num_layers=int(self.config.get("num_layers", 1)),
            output_size=int(self.config.get("output_size", 9)),
        )
        import torch
        ckpt = torch.load(self.checkpoint_path, map_location=torch.device("cpu"),
                          weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(torch.device(device))
        self.model.eval()

    def forecast(self, history: np.ndarray) -> Dict[str, Any]:
        """Forecast (lat, lon, wind) at +6/+12/+24 h from a (5,7) history array."""
        h = np.asarray(history, dtype=np.float32)
        if h.ndim != 2 or h.shape != (5, 7):
            raise ValueError(
                f"history must be shape (5, 7) with order "
                f"{FEATURES}; got {h.shape}")
        if np.isnan(h).any() or np.isinf(h).any():
            raise ValueError("history contains NaN/Inf; refusing to forecast")

        import torch
        with torch.no_grad():
            xn = np.asarray(self.normalizer.normalize_X(h), dtype=np.float32)
            xt = torch.from_numpy(xn).unsqueeze(0)
            pred = self.model(xt).squeeze(0)
            pred_denorm = self.normalizer.denormalize_Y(pred.numpy())

        if np.isnan(pred_denorm).any() or np.isinf(pred_denorm).any():
            raise RuntimeError("forecast produced NaN/Inf; aborting")

        forecasts = []
        for i, hours in enumerate(HORIZONS_HOURS):
            forecasts.append({
                "hours": int(hours),
                "latitude": float(pred_denorm[i, 0]),
                "longitude": _wrap_lon(float(pred_denorm[i, 1])),
                "wind_speed_kmh": float(pred_denorm[i, 2]),
            })
        return {"forecast": forecasts}