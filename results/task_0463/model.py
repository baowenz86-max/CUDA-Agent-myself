import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, channels, padding_size, min_value):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.min_value = min_value

    def forward(self, x):
        x = self.reflection_pad(x)
        return torch.minimum(x, self.min_value)


batch_size = 32
channels = 3
width = 1024
height = 1024
padding_size = (1, 2, 1, 2)  # (left, right, top, bottom)
min_value = torch.tensor(0.0)


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [channels, padding_size, min_value]