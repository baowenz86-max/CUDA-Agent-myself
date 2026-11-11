import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.triu = lambda x: torch.triu(x, diagonal=0)
        self.batch_norm = nn.BatchNorm2d(num_channels, eps, momentum, affine, track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.triu(x)
        x = self.batch_norm(x)
        return x


num_channels = 64
height = 512
width = 56


def get_inputs():
    x = torch.randn(64, num_channels, height, width)  # Increased batch size from 1 to 64
    return [x]


def get_init_inputs():
    return [num_channels]