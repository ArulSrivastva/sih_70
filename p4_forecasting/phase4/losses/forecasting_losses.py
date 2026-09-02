"""Loss functions for P4 Phase-4.

All criteria operate on forecasts/targets of shape (B, 3, 3) where
``[..., 0]=lat, [..., 1]=lon, [..., 2]=wind_speed`` across the +6/+12/+24 h rows.

Provided criteria:
  * ForecastMSELoss
  * ForecastHuberLoss
  * WeightedMultiTaskLoss   (track_weight, intensity_weight, horizon_weights)

All weights are explicit constructor arguments; nothing is hard-coded.
"""

from __future__ import annotations

import torch
from torch import nn

TRACK_COLS = [0, 1]   # lat, lon
INTENSITY_COL = [2]   # wind_speed
N_HORIZONS = 3


class ForecastMSELoss(nn.Module):
    """Plain MSE over all (B,3,3) cells."""

    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.MSELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)


class ForecastHuberLoss(nn.Module):
    """Huber (smooth-L1) loss over all (B,3,3) cells with configurable delta."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = float(delta)
        self._loss = nn.SmoothL1Loss(reduction="mean", beta=self.delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)


class WeightedMultiTaskLoss(nn.Module):
    """Weighted track / intensity multi-task loss with horizon weights.

    ``track_loss`` is the (horizon-weighted) mean over the 6 track cells of the
    squared (or Huber) error; ``intensity_loss`` is the (horizon-weighted) mean
    over the 3 wind cells.  The returned value is::

        out = (track_weight*track_loss + intensity_weight*intensity_loss)
              / (track_weight + intensity_weight)

    Horizon weights are normalised by their sum on each side so their absolute
    scale does not distort the track/intensity balance.
    """

    def __init__(
        self,
        track_weight: float = 1.0,
        intensity_weight: float = 1.0,
        horizon_weights: list | tuple | None = None,
        base: str = "mse",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.track_weight = float(track_weight)
        self.intensity_weight = float(intensity_weight)
        hw = list(horizon_weights) if horizon_weights is not None else [1.0, 1.0, 1.0]
        if len(hw) != N_HORIZONS:
            raise ValueError(f"horizon_weights must have length {N_HORIZONS}; got {hw}")
        self.horizon_weights = [float(w) for w in hw]
        self.base = base
        self.huber_delta = float(huber_delta)
        if base not in ("mse", "huber"):
            raise ValueError(f"base must be 'mse' or 'huber'; got {base!r}")

    def _cell_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.base == "mse":
            return (pred - target) ** 2
        delta = self.huber_delta
        err = torch.abs(pred - target)
        quad = torch.clamp(err, max=delta) ** 2 * 0.5
        lin = delta * (err - 0.5 * delta)
        return torch.where(err <= delta, quad, lin)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 3 or pred.shape[1:] != (3, 3):
            raise ValueError(f"pred must be (B,3,3); got {tuple(pred.shape)}")
        if target.shape != pred.shape:
            raise ValueError(f"target {tuple(target.shape)} != pred {tuple(pred.shape)}")

        err = self._cell_error(pred, target)
        hw = torch.as_tensor(self.horizon_weights, dtype=pred.dtype, device=pred.device)
        hw = hw / hw.sum()

        track_h = torch.mean(err[:, :, TRACK_COLS], dim=2)      # (B,3)
        intensity_h = err[:, :, INTENSITY_COL].squeeze(-1)      # (B,3)

        track_loss = torch.dot(hw, torch.mean(track_h, dim=0))
        intensity_loss = torch.dot(hw, torch.mean(intensity_h, dim=0))

        denom = self.track_weight + self.intensity_weight
        return (self.track_weight * track_loss + self.intensity_weight * intensity_loss) / denom


def build_criterion(config: dict) -> nn.Module:
    """Construct a loss module from an experiment config dict."""
    loss = str(config.get("loss", "mse")).lower()
    if loss == "mse":
        return ForecastMSELoss()
    if loss == "huber":
        return ForecastHuberLoss(delta=float(config.get("huber_delta", 1.0)))
    if loss in ("weighted", "multi-task", "multitask"):
        return WeightedMultiTaskLoss(
            track_weight=float(config.get("track_weight", 1.0)),
            intensity_weight=float(config.get("intensity_weight", 1.0)),
            horizon_weights=config.get("horizon_weights", [1.0, 1.0, 1.0]),
            base=config.get("loss_base", "mse"),
            huber_delta=float(config.get("huber_delta", 1.0)),
        )
    raise ValueError(f"unknown loss '{loss}'; expected mse|huber|weighted")