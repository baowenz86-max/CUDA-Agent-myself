import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, upscale_factor):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


batch_size = 512
in_channels = 16
out_channels = 4
height, width = 64, 64  # Increased from 32x32 to 64x64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
upscale_factor = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, upscale_factor]