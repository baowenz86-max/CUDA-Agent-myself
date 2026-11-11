import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = torch.cat([x, x], dim=-1)
        return x


batch_size = 512
channels = 3
height = 256
width = 256


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [1]