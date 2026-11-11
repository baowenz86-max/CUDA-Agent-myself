import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad_size, bias_shape):
        super().__init__()
        self.pad = nn.ConstantPad3d(pad_size, 0)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.pad(x)
        x = x + self.bias
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
num_channels = 3
height, width, depth = 64, 64, 64  # Increased spatial dimensions to increase computation
pad_size = (1, 2, 1, 2, 1, 2)  # Padding for each dimension: (left, right, top, bottom, front, back)
bias_shape = (num_channels, 1, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width, depth)]


def get_init_inputs():
    return [pad_size, bias_shape]