import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, channels, pad_size, pad_mode):
        super().__init__()
        self.pad = nn.ReflectionPad2d(pad_size)
        self.channels = channels

    def forward(self, x):
        x = self.pad(x)
        x = torch.argsort(x, dim=1)
        return x


channels = 3
pad_size = (1, 2, 1, 2)
pad_mode = "reflect"
batch_size = 512
height, width = 64, 64  # Increased spatial dimensions from 32x32 to 64x64


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [channels, pad_size, pad_mode]