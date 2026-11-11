import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, padding: tuple, bias: bool = False):
        super().__init__()
        self.pad = nn.ConstantPad2d(padding, 0)
        self.abs = torch.abs
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.abs(x)
        return self.conv2d(x)


batch_size = 32  # Increased from 16 to increase runtime
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
width = 512
height = 256
padding = (1, 2)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding]