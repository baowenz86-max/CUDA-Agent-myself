import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding_size)

    def forward(self, x):
        x = torch.ceil(x)
        x = self.pad(x)
        return x


batch_size = 256  # Increased from 16 to 256 to increase runtime
channels = 3
height = 512
width = 256
padding_size = (3, 3, 3, 3)


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [padding_size]