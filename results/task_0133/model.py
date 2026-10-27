import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, pad_size, size_before, size_after):
        super().__init__()
        self.pad1d = nn.ConstantPad1d(pad_size, value=0)
        self.size_after = size_after

    def forward(self, x):
        x = self.pad1d(x)
        x = F.interpolate(x, size=self.size_after)
        return x


batch_size = 32
in_channels = 3
out_channels = 64
pad_size = (3, 5)
width = 64
height = 64
size_after = (2048, 2048)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [pad_size, height, size_after[1]]