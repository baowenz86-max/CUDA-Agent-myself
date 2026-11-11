import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape1, input_shape2):
        super().__init__()
        self.input_shape1 = input_shape1
        self.input_shape2 = input_shape2

    def forward(self, x1, x2):
        x1 = torch.cat((x1, x2), dim=1)
        x1 = torch.clone(x1)
        return x1


batch_size = 512
channel1 = 32
channel2 = 32
height, width = 64, 64


def get_inputs():
    x1 = torch.randn(batch_size, channel1, height, width)
    x2 = torch.randn(batch_size, channel2, height, width)
    return [x1, x2]


def get_init_inputs():
    return [(batch_size, channel1, height, width), (batch_size, channel2, height, width)]