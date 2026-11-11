import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding, input_shape):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.input_shape = input_shape

    def forward(self, x):
        x = torch.acos(x)
        x = self.reflection_pad(x)
        return x


batch_size = 128
channels = 3
height, width = 512, 512  # Increased width from 32 to 512 to increase computation
padding = 3
input_shape = (batch_size, channels, height, width)


def get_inputs():
    x = torch.clamp(torch.randn(input_shape), -0.999, 0.999)
    return [x]


def get_init_inputs():
    return [padding, input_shape]