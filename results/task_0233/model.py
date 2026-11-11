import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale_factor=None, mode="nearest", align_corners=None):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def forward(self, x):
        min_indices = torch.argmin(x, dim=1)
        x = self.upsample(x)
        return x, min_indices


batch_size = 128
height, width = 512, 512  # Increased width from 32 to 512 to increase computational load
scale_factor = 2.0
mode = "bilinear"
align_corners = None


def get_inputs():
    return [torch.randn(batch_size, 1, height, width)]


def get_init_inputs():
    return [scale_factor, mode, align_corners]