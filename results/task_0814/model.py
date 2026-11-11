import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale_factor, mode, padding_dim, padding_value):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.pad = nn.ConstantPad2d(padding_dim, padding_value)

    def forward(self, x):
        x = self.upsample(x)
        x = self.pad(x)
        return x


batch_size = 256  # Increased from 16 to increase runtime
channels = 3
height = 64
width = 512
scale_factor = 2
mode = "bilinear"
padding_dim = (1, 1, 1, 1)
padding_value = 0


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [scale_factor, mode, padding_dim, padding_value]