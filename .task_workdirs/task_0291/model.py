import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x):
        x = torch.tril(x)
        return F.relu(x)


def get_inputs():
    batch_size, channels, height, width = 512, 16, 256, 256
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    batch_size, channels, height, width = 512, 16, 256, 256
    return [(batch_size, channels, height, width)]