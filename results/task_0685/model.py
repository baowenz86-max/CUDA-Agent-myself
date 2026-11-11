import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=0, keepdim=True)
        x = F.group_norm(x, num_groups=self.num_groups)
        return x


batch_size = 512
num_channels = 32
height, width = 256, 256
num_groups = 4


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_groups, num_channels]