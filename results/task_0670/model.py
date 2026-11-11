import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad_value):
        super(Model, self).__init__()
        self.padder = nn.ConstantPad2d(padding=pad_value, value=0)

    def forward(self, x):
        x = self.padder(x)
        x = torch.sum(x)
        return x


pad_value = (2, 2, 2, 2)  # pad 2 pixels on each side for height and width
batch_size = 16
channels = 3
height = 2048  # Increased height to increase runtime
width = 2048   # Increased width to increase runtime


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)  # Larger spatial dimensions to increase computation
    return [x]


def get_init_inputs():
    return [pad_value]