import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, alpha):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.hardswish = nn.Hardswish()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.avg_pool(x)
        x = self.hardswish(x)
        x = self.conv_transpose(x)
        x = self.leaky_relu(x)
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
in_channels = 3
out_channels = 16
height, width = 64, 64  # Increased spatial dimensions to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
alpha = 0.01


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, alpha]