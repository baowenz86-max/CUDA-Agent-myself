import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, interp_scale_factor, interp_mode, argmax_dim):
        super().__init__()
        self.interp_scale_factor = interp_scale_factor
        self.interp_mode = interp_mode
        self.argmax_dim = argmax_dim

    def forward(self, x):
        x = torch.round(x)
        x = F.interpolate(x, scale_factor=self.interp_scale_factor, mode=self.interp_mode)
        x = torch.argmax(x, dim=self.argmax_dim)
        return x


batch_size = 128  # Increased batch size to increase runtime
height, width = 512, 128
in_channels = 3
interp_scale_factor = 2.0
interp_mode = "bilinear"
argmax_dim = 0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, interp_scale_factor, interp_mode, argmax_dim]