import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, batch_size, height, width):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

    def forward(self, x):
        x = F.unfold(x, kernel_size=(self.height, self.width))
        x = torch.abs(x)
        return x


batch_size = 32
in_channels = 3
kernel_size = 3
width = 64
height = 128


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, batch_size, height, width]