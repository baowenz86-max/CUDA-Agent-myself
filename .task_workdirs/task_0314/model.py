import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.avg_pool(x)
        return x


batch_size = 16
in_channels = 3
out_channels = 64
height = 1536  # Increased from 512 to 1536 to increase runtime
width = 1536   # Increased from 256 to 1536 to increase runtime
kernel_size = 3
stride = 2
padding = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]