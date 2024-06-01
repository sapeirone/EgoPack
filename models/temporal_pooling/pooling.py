import torch
from torch_geometric.nn import PositionalEncoding, TemporalEncoding, Linear

from typing import Optional, Literal

import logging
logger = logging.getLogger(__name__)


class TemporalPooling(torch.nn.Module):
    """Temporal pooling layer"""

    def __init__(self, input_size: int, output_size: int, num_segments: int,
                 encoding: Optional[Literal['positional', 'temporal', 'learnt']] = None,
                 encoding_level: Literal['frame', 'action'] = 'frame') -> None:
        """Initialize the temporal pooling layer

        Encoding levels:
         - frame: encoding is applied to all the segments of an action, separately for each action
         - action: encoding is applied to all the actions inside the same batch (segments of the 
            same action share the same positional embedding)
         - video: encoding is applied to all segments inside the video (TODO)

        Parameters
        ----------
        input_size : int
            size of the input features
        output_size : int
            size of the output features
        num_segments : int
            number of action segments
        encoding : Optional[Literal["positional", "temporal", "learnt"]]
            type of positional encoding to apply to the input
        encoding_level : Literal["frame", "action"], optional
            encoding level, by default 'frame'
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_segments = num_segments

        self.encoding_level = encoding_level
        self.encoding = self._build_positional_encoding(input_size, encoding, num_segments)

        # MLP applied on top of the positional encoding
        self.encoding_mlp = None
        if self.encoding is not None:
            self.encoding_mlp = Linear(input_size, input_size)

    def _build_positional_encoding(self, input_size: int, encoding: Optional[Literal['positional', 'temporal', 'learnt']], num_segments: int):
        if encoding == 'positional':
            return PositionalEncoding(input_size)
        elif encoding == 'temporal':
            return TemporalEncoding(input_size)
        elif encoding == 'learnt':
            if self.encoding_level == 'frame':
                return torch.nn.Parameter(torch.rand((num_segments, input_size)), requires_grad=True)
            else:
                logger.warning('Learnt encoding is supported only for frame level encoding!')

        logging.warning("No positional encoding in use")
        return None

    def apply_positional_embedding(self, x: torch.Tensor, batch: torch.Tensor, pos: torch.Tensor):
        # input has shape (batch_size, num_segments, features_size)
        # batch has size (batch_size, )
        # pos has size (batch_size, )

        if self.encoding is None:
            # return x unmodified
            return x

        # frame level encoding
        if self.encoding_level == 'frame':
            if isinstance(self.encoding, torch.nn.Parameter):
                return x + self.encoding_mlp(self.encoding.unsqueeze(0))  # unsqueeze on the segments dimension
            return x + self.encoding_mlp(self.encoding(torch.arange(0, self.num_segments, device=x.device, dtype=torch.float32)))

        # video level encoding
        data = torch.zeros_like(x)
        for b in batch.unique():
            data[batch == b] = x[batch == b] + self.encoding_mlp(self.encoding(pos[batch == b]).unsqueeze(1))

        return data

    def forward(self, x: torch.Tensor, batch: torch.Tensor, pos: torch.Tensor):
        # input has shape (batch_size, num_segments, features_size)
        # batch has size (batch_size, )
        # pos has size (batch_size, )
        raise NotImplementedError("TemporalPooling.forward is not implemented")


if __name__ == '__main__':
    bs, n_segments, feat_size = 32, 8, 1024
    video_size = 4
    x = torch.rand((bs, n_segments, feat_size))
    batch = torch.arange(0, bs // video_size).repeat_interleave(video_size)
    pos = torch.arange(0, video_size).repeat(bs // video_size)

    print(batch)
    print(pos)

    pooling = TemporalPooling(feat_size, feat_size, n_segments, 'positional', 'video')
    pooling.apply_positional_embedding(x, batch, pos)
