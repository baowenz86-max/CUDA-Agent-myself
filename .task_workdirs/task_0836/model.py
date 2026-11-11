import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upscale_factor):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.conv1d(
            x, self.conv1d.weight, bias=self.conv1d.bias, stride=self.conv1d.stride, padding=self.conv1d.padding
        )
        x = self.pixel_shuffle(x)
        return x


batch_size = 128
in_channels = 32
out_channels = 128
seq_length = 4096  # Increased from 512 to 4096 to increase runtime
kernel_size = 5
stride = 1
padding = 2
upscale_factor = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, seq_length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, upscale_factor]