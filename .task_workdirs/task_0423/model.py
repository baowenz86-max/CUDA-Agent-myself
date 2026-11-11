import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bias_shape):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = torch.clamp(x, min=0.0, max=1.0)
        x = torch.triu(x)
        x = x + self.bias
        return x


batch_size = 128
height, width = 1024, 1024
bias_shape = (height, width)


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return [bias_shape]