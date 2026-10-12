import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.pad = nn.ReflectionPad2d(pad_size)

    def forward(self, x):
        x = torch.median(x, dim=0).values
        grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, x.size(0)), torch.linspace(0, 1, x.size(1)), indexing="ij")
        x = self.pad(x)
        return x


batch_size = 128
in_channels = 3
height, width = 512, 256  # Increased width to 256 to increase runtime
pad_size = 4


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [pad_size]