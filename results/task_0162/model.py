import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.leaky_relu(x, negative_slope=0.01)
        x = torch.relu(x)
        return x


batch_size = 16
in_channels = 3
height = 2560
width = 2560


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return []