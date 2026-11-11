import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, start, end, step, kernel_size, stride):
        super().__init__()
        self.arange = torch.arange(start, end, step)
        self.pool = nn.AvgPool2d(kernel_size, stride=stride)

    def forward(self, x):
        return self.pool(x)


batch_size = 32
channels = 16
height = 1024  # Increased from 512
width = 256    # Increased from 128
kernel_size = (2, 2)
stride = 2


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [0, 10, 1, kernel_size, stride]