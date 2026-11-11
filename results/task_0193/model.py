import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=num_channels, num_channels=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=1, keepdim=True)
        return x


num_channels = 3
batch_size = 512
height = 1024
width = 1024


def get_inputs():
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    return [num_channels]