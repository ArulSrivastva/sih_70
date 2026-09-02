"""P4 Phase-4 loss functions package."""

from .forecasting_losses import (
    ForecastHuberLoss,
    ForecastMSELoss,
    WeightedMultiTaskLoss,
    build_criterion,
)

__all__ = [
    "ForecastHuberLoss",
    "ForecastMSELoss",
    "WeightedMultiTaskLoss",
    "build_criterion",
]