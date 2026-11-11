import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.channel_shuffle = nn.ChannelShuffle(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_shuffle(x)


batch_size = 32  # Increased from 16 to 32 to increase runtime
in_channels = 64
height = 512
width = 256


def get_inputs():
    # Assuming the input tensor has shape [batch_size, in_channels, height, width]
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    # The groups parameter depends on the structure of your model.
    # It should be the number of groups that you want to shuffle your channels into.
    # Here we set it to 2 for demonstration purposes.
    return [2]