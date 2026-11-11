import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return torch.pixel_unshuffle(x, self.upscale_factor)


batch_size = 64
channels = 3
height = 1024
width = 1024
upscale_factor = 2


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [upscale_factor]