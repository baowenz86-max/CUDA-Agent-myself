import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, scale_factor=1.0, mode="bilinear"):
        super(Model, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = torch.where(x > 0, x, torch.zeros_like(x))
        x = nn.Upsample(scale_factor=self.scale_factor, mode=self.mode)(x)
        return x


batch_size = 16
channels = 3
height = 512
width = 512  # Increased from 224 to 512 to increase computation
scale_factor = 2.0
mode = "bilinear"


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [scale_factor, mode]