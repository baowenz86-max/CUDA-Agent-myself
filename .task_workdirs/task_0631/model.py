import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.log10(x)
        return x


batch_size = 16
channels = 3
height = 4096
width = 4096


def get_inputs():
    return [torch.rand(batch_size, channels, height, width)]


def get_init_inputs():
    return []