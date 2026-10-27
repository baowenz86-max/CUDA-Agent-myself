import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_channels, upscale_factor, pad_size, reflect_padding_dims, upscale_mode):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.pad_size = pad_size
        self.reflect_padding_dims = reflect_padding_dims
        self.upscale_mode = upscale_mode

        self.reflection_pad = nn.ReflectionPad2d(padding=self.reflect_padding_dims)
        self.upscale = nn.Upsample(scale_factor=self.upscale_factor, mode=self.upscale_mode)

    def forward(self, x):
        x = self.reflection_pad(x)
        non_zero_elements = torch.count_nonzero(x)
        x = self.upscale(x)
        return non_zero_elements, x


batch_size = 8
input_channels = 3
upscale_factor = 2
pad_size = (1, 1, 1, 1)
reflect_padding_dims = (1, 1, 1, 1)
upscale_mode = "bilinear"
height = 1024  # Increased from 512 to increase runtime
width = 1024   # Increased from 512 to increase runtime


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [input_channels, upscale_factor, pad_size, reflect_padding_dims, upscale_mode]