import torch
import torch.nn as nn

from typing import Literal

TaskLiteral = Literal['ar', 'oscc', 'lta', 'pnr', 'ant']


class ProjectionTask(torch.nn.Module):

    def __init__(self, name: str, input_size: int, features_size: int = 1024, dropout: float = 0):
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.features_size = features_size

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, features_size),
            nn.LayerNorm(features_size),
            nn.ReLU(),
            nn.Linear(features_size, features_size),
        )

    def forward_features(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.net(x)
