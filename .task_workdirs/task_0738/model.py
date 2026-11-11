import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = torch.stack((x, y), dim=0)
        x = torch.clone(x)
        return x


batch_size = 512
channels = 3
height, width = 256, 256


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)
    return [x, y]


def get_init_inputs():
    return []