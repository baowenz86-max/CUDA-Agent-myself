import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU()
        self.conv_transpose3d = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.adaptive_avg_pool(x)
        x = self.conv3d(x)
        x = self.relu(x)
        x = x + self.bias
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.conv_transpose3d(x)
        return x


batch_size = 512
in_channels = 3
out_channels = 64  # Increased from 16 to increase computation
height, width, depth = 128, 128, 128  # Increased from 32 to increase input volume
kernel_size = (3, 3, 3)
stride = (1, 1, 1)  # Changed from (2,2,2) to process more elements
padding = (1, 1, 1)
output_padding = (0, 0, 0)  # Adjusted for stride=1
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]