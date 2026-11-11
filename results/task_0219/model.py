import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        x = torch.mean(x, dim=self.in_channels, keepdim=True)
        x = F.softplus(x)
        return x


batch_size = 256  # Increased from 16 to increase computation
in_channels = 3
height = 1024  # Increased from 512 to increase computation
width = 1024  # Increased from 256 to increase computation


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels]