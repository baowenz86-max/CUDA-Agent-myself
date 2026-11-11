import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super().__init__()
        self.equal_op = torch.equal
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


batch_size = 16
in_channels = 3
height = 3072  # Increased from 512
width = 1536   # Increased from 256
kernel_size = 2
stride = 2


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride]