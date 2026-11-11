import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.reflection_pad = nn.ReflectionPad2d((1, 1, 1, 1))  # Adjust padding values as needed

    def forward(self, x):
        x = self.upsample(x)
        x = self.reflection_pad(x)
        return x


batch_size = 16
height, width = 1024, 1024  # Increased spatial dimensions to increase computation
scale_factor = 2.0
mode = "bilinear"


def get_inputs():
    return [torch.randn(batch_size, 1, height, width)]


def get_init_inputs():
    return [scale_factor, mode]