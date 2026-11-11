import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        x = F.normalize(x, p=2.0, dim=1, eps=1e-5)
        x = self.conv_transpose(x)
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 64, 64  # Increased from 32x32 to 64x64 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]