import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upscale_factor):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels * upscale_factor**2, kernel_size)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.pixel_shuffle(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
width = 8192  # Increased from 512 to 8192 to increase runtime
kernel_size = 3
upscale_factor = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, upscale_factor]