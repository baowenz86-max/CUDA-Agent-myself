import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.selu(x)
        x = torch.einsum("bwh->bwh", x)
        return x


batch_size = 16
height = 5120
width = 5120


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return []