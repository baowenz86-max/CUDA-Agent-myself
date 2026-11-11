import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int, pool_size: tuple, min_val: float, max_val: float):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size)
        self.hardtanh = nn.Hardtanh(min_val=min_val, max_val=max_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = self.hardtanh(x)
        return x


batch_size = 16
in_channels = 3
height = 2048
width = 2048
pool_size = (2, 2)
min_val = -1.0
max_val = 1.0


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, pool_size, min_val, max_val]