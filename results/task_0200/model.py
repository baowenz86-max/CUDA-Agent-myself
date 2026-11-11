import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        x = F.hardshrink(x, lambd=0.5)
        x = F.elu(x)
        return x


channels = 64
height, width = 512, 512


def get_inputs():
    x = torch.randn(16, channels, height, width)
    return [x]


def get_init_inputs():
    return [channels]