"""GRU cyclone forecaster for P4 Phase-4 (locked contract: 16 -> GRU -> 9).

    (batch, 5, 16)  ->  GRU(16 -> hidden_size, num_layers, dropout)
                    ->  last hidden state -> Linear(hidden_size -> 9)
                    ->  view (batch, 3, 3)   [lat, lon, wind] x horizons
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


class GRUCyclone(nn.Module):
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

        gru_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-2:] != (5, self.input_size):
            raise ValueError(
                f"expected history (batch,5,{self.input_size}); got {tuple(history.shape)}")
        out, _ = self.gru(history)
        last = out[:, -1, :]
        return self.head(last).view(-1, N_HORIZONS, N_TARGETS)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_numpy(self, history, device: str = "cpu") -> Tuple[torch.Tensor, np.ndarray]:
        if isinstance(history, np.ndarray):
            history = torch.from_numpy(np.ascontiguousarray(history)).to(
                dtype=torch.float32, device=torch.device(device))
        with torch.inference_mode():
            out = self.forward(history)
        return out, (out.numpy() if out.device.type == "cpu" else out.cpu().numpy())