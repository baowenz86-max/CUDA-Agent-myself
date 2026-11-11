import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        x = torch.abs(x)
        x = F.pad(x, (1, 2, 3, 4))
        return x


batch_size = 16
in_channels = 3
height = 2560  # Increased from 512
width = 1280   # Increased from 256


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels]