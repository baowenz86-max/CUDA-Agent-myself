import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = torch.relu(x)
        return x


batch_size = 16
c, h, w = 3, 1024, 1024  # Increased height and width from 512 to 1024
scale_factor = 2.0
mode = "bilinear"


def get_inputs():
    return [torch.randn(batch_size, c, h, w)]


def get_init_inputs():
    return [scale_factor, mode]