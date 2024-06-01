import torch.nn as nn
from einops import rearrange

from models.temporal_pooling.pooling import TemporalPooling

import logging
logger = logging.getLogger(__name__)


class TRNPooling(TemporalPooling):

    def __init__(
        self,
        input_size: int = 1024,
        output_size: int = 1024,
        num_segments: int = 8,
        hidden_size: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_size, output_size, num_segments)

        self.input_size = input_size
        self.num_segments = num_segments

        logging.info(f"Instantiating TRNPooling with parameters: input_size={input_size}, "
                     f"hidden_size={hidden_size}, output_size={output_size}, num_segments={num_segments}, dropout={dropout}")

        self.proj = nn.Sequential(
            # first layer: self.num_segments * input_size -> hidden_size
            nn.Linear(self.num_segments * input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # second layer: hidden_size -> hidden_size
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # final projection: hidden_size -> output_size
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, *_):
        x = rearrange(x, 'bs segments h -> bs (segments h)', segments=self.num_segments, h=self.input_size)
        return self.proj(x)
