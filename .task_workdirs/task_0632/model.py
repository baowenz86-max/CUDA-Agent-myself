import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, reduction_dims, scale_factor, mode):
        super().__init__()
        self.reduction_dims = reduction_dims
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = torch.sum(x, dim=self.reduction_dims)
        return x


batch_size = 16
channels = 3
height = 4096
width = 2048
reduction_dims = (2, 3)
scale_factor = 0.5
mode = "bilinear"


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [reduction_dims, scale_factor, mode]