import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.abs(x)
        x = torch.tan(x)
        return x


batch_size = 512
channels = 3
height = 384
width = 384


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []