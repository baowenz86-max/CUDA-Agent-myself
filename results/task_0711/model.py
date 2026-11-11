import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, upscale_factor, channels_factor):
        super(Model, self).__init__()
        self.sum_operation = torch.sum
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.channels_factor = channels_factor

    def forward(self, x):
        x = self.sum_operation(x, dim=1, keepdim=True)  # torch.sum over channels
        x = x.expand(-1, self.channels_factor, -1, -1)  # replicate sum to C' channels
        x = self.pixel_shuffle(x)  # pixel shuffle
        return x


batch_size = 512  # Increased to ensure first dimension is >=512
channels = 16
height = 256  # Increased spatial dimension to increase runtime
width = 256   # Increased spatial dimension to increase runtime
upscale_factor = 2


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [upscale_factor, channels // (upscale_factor**2)]