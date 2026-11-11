import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def forward(self, x):
        x = self.upsample(x)
        x = torch.ceil(x)
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
in_channels = 3
height, width = 128, 128  # Increased spatial dimensions to increase computation
scale_factor = 2.0
mode = "bilinear"
align_corners = False


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [scale_factor, mode, align_corners]