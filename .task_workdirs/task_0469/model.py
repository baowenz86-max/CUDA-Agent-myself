import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding_size)

    def forward(self, x):
        x = self.pad(x)
        x = torch.fmod(x, 1.0)
        return x


batch_size = 16
channels = 3
height = 2048  # Increased from 512
width = 1024   # Increased from 256
padding_size = (2, 2, 1, 1)


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [padding_size]