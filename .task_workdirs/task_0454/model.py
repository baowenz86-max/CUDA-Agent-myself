import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int, pooling_size: int):
        super().__init__()
        self.abs = torch.abs
        self.adaptive_pool = nn.AdaptiveAvgPool1d(pooling_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.abs(x)
        x = self.adaptive_pool(x)
        return x


batch_size = 16
in_channels = 64
width = 102400  # Increased from 512 to 102400 to increase computation
pooling_size = 16


def get_inputs():
    x = torch.randn(batch_size, in_channels, width)
    return [x]


def get_init_inputs():
    return [in_channels, pooling_size]