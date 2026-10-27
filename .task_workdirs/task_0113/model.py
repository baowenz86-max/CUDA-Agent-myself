import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.argmax(x, dim=1)


batch_size = 16
channels = 3
height = 4096  # Increased from 512
width = 1024   # Increased from 128


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []