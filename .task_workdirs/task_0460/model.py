import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        # Unbind the tensor into a list of tensors along a specified dimension
        x = torch.unbind(x, dim=1)
        # Apply pixel shuffle operation
        x = [self.pixel_shuffle(item) for item in x]
        # Stack the results back together along the first dimension
        x = torch.stack(x, dim=1)
        return x


batch_size = 512
channels = 32
height = 128
width = 128
upscale_factor = 1


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [upscale_factor]