import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, start, end, step):
        super().__init__()
        self.sequence = torch.arange(start, end, step)

    def forward(self, x):
        x = torch.relu(x)
        return x


batch_size = 64
channels = 3
height = 2048
width = 2048


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [0, 10, 0.1]