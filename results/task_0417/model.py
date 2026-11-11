import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x, _ = F.max_pool2d_with_indices(
            x, kernel_size=self.max_pool.kernel_size, stride=self.max_pool.stride, padding=self.max_pool.padding
        )
        return x


batch_size = 16
in_channels = 3
out_channels = 64
height = 1024  # Increased from 512 to 1024
width = 1024   # Increased from 256 to 1024
kernel_size = (3, 5)
stride = 1
padding = (1, 2)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]