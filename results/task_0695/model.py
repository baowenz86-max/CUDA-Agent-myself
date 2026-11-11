import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad, dim):
        super().__init__()
        self.pad = nn.ConstantPad2d(pad, 0)
        self.dim = dim

    def forward(self, x):
        x = self.pad(x)
        x = torch.cat([x, x], dim=self.dim)
        return x


pad = (1, 2, 1, 2)
dim = 1
batch_size = 16
channels = 3
height = 1280  # Increased to increase computational load
width = 1280   # Increased to increase computational load


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [pad, dim]