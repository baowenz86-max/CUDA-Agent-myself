import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(x)


batch_size = 512
channels = 12
height = 128
width = 128
upscale_factor = 3


def get_inputs():
    return [torch.randn(batch_size, channels * upscale_factor * upscale_factor, height, width)]


def get_init_inputs():
    return [upscale_factor]