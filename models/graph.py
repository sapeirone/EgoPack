import hydra

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from torch_geometric.data import Data

import logging


logger = logging.getLogger(__name__)


class Graph(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 1024,
        depth: int = 3,
        pre_dropout: float = 0,
        temporal_pooling=None,
        num_segments: int = 8,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.num_segments = num_segments
        self.pre_dropout = torch.nn.Dropout(pre_dropout)

        if temporal_pooling:
            self.temporal_pooling = hydra.utils.instantiate(temporal_pooling, input_size, hidden_size, num_segments)
        else:
            self.temporal_pooling = None

        self.positional_encoding = gnn.PositionalEncoding(hidden_size)

        if depth > 0:
            self.conv_layers = []
            for _ in range(depth):
                self.conv_layers.append((gnn.SAGEConv(hidden_size, hidden_size, project=True), "x, edges -> x"))
                self.conv_layers.append((gnn.LayerNorm(hidden_size), "x -> x"))
                self.conv_layers.append((nn.LeakyReLU(negative_slope=0.2), "x -> x"))

            self.conv_layers.append((nn.Linear(hidden_size, hidden_size), "x -> x"))

            self.net = gnn.Sequential("x, edges, batch", self.conv_layers)

    def configure_optimizers(self, _):
        return self.parameters()

    def forward(self, data: Data, *args, **kwargs):
        x = data.x

        x = self.pre_dropout(x)

        # apply a positional or temporal encoding
        if self.temporal_pooling:
            x = self.temporal_pooling(x, data.batch, data.pos)

        if hasattr(self, "net"):
            x = x + self.net(x + self.positional_encoding(data.pos), data.edge_index, data.batch)

        return x
