import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        x = torch.acos(x)
        x = self.conv_transpose(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
length = 16384  # Increased from 512 to 16384 to increase runtime
kernel_size = 3
stride = 2
padding = 1
output_padding = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    x = torch.clamp(x, -1.0, 1.0)  # Ensure input is within the valid range for torch.acos
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]