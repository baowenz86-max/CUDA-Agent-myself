import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.softplus = nn.Softplus()
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)

    def forward(self, x):
        x = self.softplus(x)
        x = self.avg_pool(x)
        return x


batch_size = 512
in_channels = 3
height, width = 256, 256  # Increased from 224x224 to increase runtime
kernel_size = 3
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, pool_kernel_size, pool_stride, pool_padding]