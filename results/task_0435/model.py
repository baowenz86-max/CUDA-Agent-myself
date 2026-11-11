import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_unshuffle(x)


batch_size = 16
channels = 128  # Increased from 3 to 128 to increase computation load
height = 512
width = 256
upscale_factor = 2


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [upscale_factor]