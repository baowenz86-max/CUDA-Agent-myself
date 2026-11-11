import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.relu(x)
        x = torch.abs(x)
        return x


batch_size = 80
height, width = 1024, 1024


def get_inputs():
    x = torch.randn(batch_size, 3, height, width)
    return [x]


def get_init_inputs():
    return []