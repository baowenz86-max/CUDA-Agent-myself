import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.abs(x)
        x = nn.ReLU6()(x)
        return x


batch_size = 256
width = 1024
height = 1024


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return []