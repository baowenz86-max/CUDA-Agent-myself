import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.roll(x, shifts=1, dims=2)
        return x


batch_size = 16
channels = 3
height = 4096
width = 1024


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []