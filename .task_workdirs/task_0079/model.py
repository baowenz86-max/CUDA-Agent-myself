import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = torch.mul(x, x)
        return x


batch_size = 128
channels = 3
height = 512
width = 512  # Increased from 224 to 512 to increase input size
padding_size = (3, 3, 3, 3)


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [padding_size]