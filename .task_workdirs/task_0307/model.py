import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, upscale_factor, mode="bilinear", align_corners=None):
        super().__init__()
        self.gelu = F.gelu
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode, align_corners=align_corners)

    def forward(self, x):
        x = self.gelu(x)
        x = self.pixel_shuffle(x)
        x = self.upsample(x)
        return x


batch_size = 512
height, width = 64, 64  # Increased from 32x32 to 64x64 to increase computation
channels = 16
upscale_factor = 2
mode = "bilinear"
align_corners = None


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [upscale_factor, mode, align_corners]