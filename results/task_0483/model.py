import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)  # Apply torch.t
        x = self.avgpool(x)
        return x


batch_size = 256
in_channels = 1024
out_channels = 512
kernel_size = 5
width = 4096  # Increased to increase computational load


def get_inputs():
    x = torch.randn(batch_size, in_channels, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]