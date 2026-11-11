import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, scaling_factor, mode="bilinear", align_corners=False):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = torch.ceil(x * self.scaling_factor)
        x = F.interpolate(x, scale_factor=self.scaling_factor, mode=self.mode, align_corners=self.align_corners)
        return x


batch_size = 16
channels = 3
height, width = 1536, 768  # Increased spatial dimensions to increase computation
scaling_factor = 2.0


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [scaling_factor]