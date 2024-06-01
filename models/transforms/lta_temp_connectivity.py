import torch

import torch_geometric.nn as nn

from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms.remove_duplicated_edges import RemoveDuplicatedEdges
from torch_geometric.data import Data

from math import floor


class LTATemporalConnectivity(BaseTransform):

    def __init__(
        self,
        r: float,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = 'source_to_target',
        num_workers: int = 1,
    ):
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.num_workers = num_workers

        self.remove_duplicates = RemoveDuplicatedEdges()

    def __call__(self, data: Data) -> Data:
        if data.batch is not None:
            raise ValueError("This transform expects no batched graphs.")

        data.edge_attr = None
        batch = data.batch if 'batch' in data else None

        data.edge_index = nn.radius_graph(
            data.pos,
            self.r,
            batch,
            self.loop,
            max_num_neighbors=self.max_num_neighbors,
            flow=self.flow,
            num_workers=self.num_workers,
        )

        # count the number of input unlabeled clips
        num_input_clips = (data.y[:, 0] == -1).sum()
        # count the number of output (to be forecasted) clips
        num_forecast_clips = (data.y[:, 0] > 0).sum()

        src = torch.arange(max((num_input_clips - self.r).ceil(), 0), num_input_clips, dtype=torch.long).repeat_interleave(num_forecast_clips)
        tgt = torch.arange(num_input_clips, num_input_clips + num_forecast_clips, dtype=torch.long).repeat(min(floor(self.r), num_input_clips))

        data.edge_index = torch.cat([torch.stack([src, tgt]), data.edge_index], dim=-1)
        return self.remove_duplicates(data)


if __name__ == '__main__':
    data = Data(x=torch.randn((32, 12)), pos=torch.arange(32).reshape(-1, 1))
    transform = LTATemporalConnectivity(r=1.5)
    data = transform(data)
