import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        x = torch.sqrt(x)
        x = torch.round(x)
        x = self.conv_transpose(x)
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
in_channels = 3
out_channels = 16
height, width = 40, 40  # Increased spatial dimensions to increase runtime
kernel_size = 3
stride = 2
padding = 1
output_padding = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width) ** 2]  # Ensure input is non-negative for sqrt


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]