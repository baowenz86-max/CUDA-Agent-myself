import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        x = self.flatten(x)
        x = x.unsqueeze(1)  # Add a channel dimension
        x = F.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            stride=self.conv_transpose1d.stride,
            padding=self.conv_transpose1d.padding,
            output_padding=self.conv_transpose1d.output_padding,
        )
        return x


batch_size = 512
in_channels = 1
out_channels = 16
input_height, input_width = 64, 64  # Increased from 8x8 to 64x64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, input_height, input_width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]