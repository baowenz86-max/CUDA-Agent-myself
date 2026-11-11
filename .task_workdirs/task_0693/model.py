import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_channels, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, input_channels * (upscale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = torch.cat((x, x), dim=1)  # Assuming the input tensor has only two channels
        x = self.pixel_shuffle(x)
        return x


batch_size = 512
input_channels = 2
height, width = 128, 128  # Increased spatial dimensions to 128x128
upscale_factor = 2


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [input_channels, upscale_factor]