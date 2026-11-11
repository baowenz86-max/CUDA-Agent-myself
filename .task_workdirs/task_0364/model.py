import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.relu(x)
        x = torch.cumsum(x, dim=1)
        return x


batch_size = 512
height, width = 32, 32


def get_inputs():
    x = torch.randn(batch_size, 512, height, width)  # Increased channels from 3 to 512
    return [x]


def get_init_inputs():
    return []