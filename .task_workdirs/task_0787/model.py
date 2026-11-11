import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, padding_size, padding_value):
        super().__init__()
        self.softmax = F.softmax
        self.pad = nn.ConstantPad2d(padding_size, padding_value)

    def forward(self, x):
        x = self.softmax(x, dim=1)
        x = self.pad(x)
        return x


batch_size = 16
in_channels = 64  # Increased from 3 to increase softmax computation
out_channels = 64
height = 512
width = 512  # Increased from 256 to increase spatial dimensions
padding_size = (1, 2)
padding_value = 0.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, padding_size, padding_value]