import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int, output_size: int):
        super().__init__()
        self.adaptive_max_pool1d = nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adaptive_max_pool1d(x)


batch_size = 16
in_channels = 64
output_size = 32
length = 100000  # Increased to increase computation time


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, output_size]