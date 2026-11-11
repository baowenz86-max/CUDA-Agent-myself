import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, padding_size, bias_shape):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding_size)
        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.pad(x)
        x = self.relu(x)
        x = torch.add(x, self.bias)
        return x


batch_size = 128
in_channels = 3
height, width = 512, 512  # Increased width from 32 to 512 to increase computation
padding_size = (1, 2, 1, 2)
bias_shape = (in_channels, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, padding_size, bias_shape]