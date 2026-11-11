import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding_size, divisor):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.reflection_pad(x)
        x = torch.remainder(x, self.divisor)
        return x


batch_size = 32
channels = 3
height = 1024
width = 1024
padding_size = (1, 2, 1, 2)
divisor = 5


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [padding_size, divisor]