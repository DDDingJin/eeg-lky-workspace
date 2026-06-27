from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FCNNBaseline(nn.Module):
    def __init__(
        self,
        *,
        num_hidden: int = 3,
        dropout_rate: float = 0.45,
        input_length: int = 50,
        num_input_channels: int = 64,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.num_input_channels = int(num_input_channels)
        units = np.round(np.linspace(1, self.input_length * self.num_input_channels, num_hidden + 2)[::-1]).astype(int)
        self.layers = nn.ModuleList([nn.Linear(int(units[i]), int(units[i + 1])) for i in range(len(units) - 1)])
        self.activations = nn.ModuleList([nn.Tanh() for _ in range(len(units) - 2)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_rate) for _ in range(len(units) - 2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        for index, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[index](x)
            x = self.dropouts[index](x)
        x = self.layers[-1](x)
        return x.flatten()
