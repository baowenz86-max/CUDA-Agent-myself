import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, groups, block_size):
        super().__init__()
        self.num_channels = num_channels
        self.groups = groups
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.channel_shuffle(x, self.groups)


def get_inputs():
    batch_size = 16
    num_channels = 64
    width = 512  # Increased from 256 to 512 to increase computation
    height = 512
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    num_channels = 64
    groups = 2
    block_size = 2
    return [num_channels, groups, block_size]