"""Lightweight multivariate LSTM for cyclone forecasting (P4 Phase 3).

Architecture:

    (batch, 5, 7)                       history: 5 steps x 7 features
        |
        v
    LSTM(7 -> 64, num_layers=1, batch_first=True, dropout=0)
        |
        v                       last hidden state over the 5-step sequence
    Linear(64 -> 9)
        |
        v
    reshape (batch, 3, 3)       (3 horizons x 3 targets)

Feature order:  [lat, lon, wind_speed, pressure, sst, wind_u, wind_v]
Target  order:  per horizon [lat, lon, wind_speed]; horizons +6h/+12h/+24h.

Small and explainable by design. No attention, no transformer, no ensembles.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 9  # 3 horizons x 3 targets
N_HORIZONS = 3
N_TARGETS = 3


class CycloneLSTM(nn.Module):
    """Predict cyclone (lat, lon, wind) at +6h/+12h/+24h from 5-step history."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        output_size: int = OUTPUT_SIZE,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (batch, 5, 7). Returns: (batch, 3, 3) forecast."""
        if history.ndim != 3 or history.shape[-2:] != (5, self.input_size):
            raise ValueError(
                f"expected history of shape (batch, 5, {self.input_size}); got {tuple(history.shape)}")
        lstm_out, _ = self.lstm(history)
        last = lstm_out[:, -1, :]          # (batch, hidden_size)
        flat = self.head(last)             # (batch, 9)
        return flat.view(-1, N_HORIZONS, N_TARGETS)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_numpy(
        self,
        history: "torch.Tensor | np.ndarray",
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, "np.ndarray"]:
        """Convenience: accept numpy/torch history, return torch + numpy forecasts."""
        import numpy as np

        if isinstance(history, np.ndarray):
            history = torch.from_numpy(np.ascontiguousarray(history)).to(
                dtype=torch.float32, device=torch.device(device))
        with torch.inference_mode():
            out = self.forward(history)
        return out, out.numpy() if out.device.type == "cpu" else out.cpu().numpy()