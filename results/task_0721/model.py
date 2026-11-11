import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x, y):
        x = torch.abs(x)
        x = torch.bitwise_xor(x.type(torch.int64), y.type(torch.int64)).type(torch.float32)
        return x


batch_size = 512
height, width = 320, 320  # Increased from 64x64 to 320x320


def get_inputs():
    x = torch.randn(batch_size, height, width)
    y = torch.randn(batch_size, height, width)
    return [x, y]


def get_init_inputs():
    input_shape = (batch_size, height, width)
    return [input_shape]