import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size, scale_factor, mode, align_corners):
        super().__init__()
        self.input_size = input_size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = torch.norm(x, p=2, dim=1, keepdim=True)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        is_complex = torch.is_complex(x)
        return x, is_complex


batch_size = 128
channels = 3
height, width = 512, 512  # Increased width from 64 to 512
scale_factor = 2.0
mode = "bilinear"
align_corners = False


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [batch_size, scale_factor, mode, align_corners]