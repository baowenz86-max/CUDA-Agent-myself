import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


batch_size = 16
channels = 3
width = 1536  # Increased from 256 (6x)
height = 3072  # Increased from 512 (6x)
kernel_size = 2
stride = 2


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [channels, kernel_size, stride]