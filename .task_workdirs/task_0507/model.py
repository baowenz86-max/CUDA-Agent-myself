import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        x = self.avgpool(x)
        return x


batch_size = 32
in_channels = 3
out_channels = 64
kernel_size = 3
stride = 2
width = 1024
height = 1024


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride]