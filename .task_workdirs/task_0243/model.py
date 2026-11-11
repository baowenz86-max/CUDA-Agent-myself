import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, avg_pool_kernel, conv_padding):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(avg_pool_kernel)
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=conv_padding)

    def forward(self, x):
        x = self.avg_pool(x)
        return self.conv3d(x)


batch_size = 2  # Increased from 1 to 2 to increase runtime
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
width = 512
height = 64
depth = 64
stride = 1
padding = (1, 2, 2)
avg_pool_kernel = (2, 2, 2)
conv_padding = (1, 2, 2)


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, avg_pool_kernel, conv_padding]