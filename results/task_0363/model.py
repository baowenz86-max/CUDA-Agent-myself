import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return F.pixel_unshuffle(x, self.upscale_factor)


batch_size = 512
channels = 3
height = 512
width = 512
upscale_factor = 2


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [upscale_factor]