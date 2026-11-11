import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x):
        x = torch.abs(x)
        x = torch.sign(x)
        return x


batch_size = 1024
channels = 3
height = 512
width = 256


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [get_inputs()[0].shape]