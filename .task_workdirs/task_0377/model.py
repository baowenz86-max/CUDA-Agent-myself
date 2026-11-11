import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.max_pool3d = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)

    def forward(self, x):
        x = self.log_softmax(x)
        x = self.max_pool3d(x)
        return x


batch_size = 16
in_channels = 8
depth, height, width = 3, 1024, 256  # Increased height and width
pool_kernel_size = (1, 3, 3)
pool_stride = (1, 2, 2)
pool_padding = (0, 1, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, pool_kernel_size, pool_stride, pool_padding]