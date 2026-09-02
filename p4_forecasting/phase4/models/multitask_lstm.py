"""Multi-task LSTM for P4 Phase-4.

Shares one recurrent (LSTM) encoder and uses two task heads:

    (batch, 5, 16)
        |
        v            shared LSTM encoder -> last hidden state
    +--------------------------+--------------------------+
    |  track head (lat, lon)   |  intensity head (wind)   |
    |  Linear(hidden -> 6)     |  Linear(hidden -> 3)     |
    |   -> view (b,3,2)        |   -> view (b,3,1)        |
    +--------------------------+--------------------------+
        |
        v  concat along last axis -> tensors
        (batch, 3, 3)   [..., 0] = latitude
                        [..., 1] = longitude
                        [..., 2] = wind_speed
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn

INPUT_SIZE = 16
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROP_OUT = 0.0
OUTPUT_SIZE = 9
N_HORIZONS = 3
N_TARGETS = 3


class MultiTaskLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        output_size: int = OUTPUT_SIZE,
        dropout: float = DROP_OUT,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.output_size = int(output_size)
        self.dropout = float(dropout)

        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        # 3 horizons * 2 track targets (lat, lon)
        self.track_head = nn.Linear(self.hidden_size, N_HORIZONS * 2)
        # 3 horizons * 1 intensity target (wind)
        self.intensity_head = nn.Linear(self.hidden_size, N_HORIZONS * 1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-2:] != (5, self.input_size):
            raise ValueError(
                f"expected history (batch,5,{self.input_size}); got {tuple(history.shape)}")
        out, _ = self.encoder(history)
        last = out[:, -1, :]
        track = self.track_head(last).view(-1, N_HORIZONS, 2)           # lat, lon
        intensity = self.intensity_head(last).view(-1, N_HORIZONS, 1)   # wind
        return torch.cat([track, intensity], dim=-1)                     # (b,3,3)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_numpy(self, history, device: str = "cpu") -> Tuple[torch.Tensor, np.ndarray]:
        if isinstance(history, np.ndarray):
            history = torch.from_numpy(np.ascontiguousarray(history)).to(
                dtype=torch.float32, device=torch.device(device))
        with torch.inference_mode():
            out = self.forward(history)
        return out, (out.numpy() if out.device.type == "cpu" else out.cpu().numpy())