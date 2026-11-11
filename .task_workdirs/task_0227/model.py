import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.hardshrink = nn.Hardshrink()
        self.reduction = reduction

    def forward(self, x):
        return self.hardshrink(x)


batch_size = 16
channels = 3
height = 4096
width = 4096


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return []