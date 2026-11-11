import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-5):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm(x)


batch_size = 16
num_channels = 64
num_groups = 32
height = 512
width = 512  # Increased from 64 to 512 to increase computation


def get_inputs():
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    return [num_channels, num_groups]