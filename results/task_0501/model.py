import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(x)


batch_size = 16
in_channels = 3
height = 2048
width = 2048
kernel_size = (3, 3)
stride = 2
padding = (1, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]