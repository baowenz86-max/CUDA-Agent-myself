import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.avg_pool(x)
        return torch.min(x, dim=1)[0]


batch_size = 128  # Increased from 16 to 128 to increase runtime
in_channels = 3
out_channels = 64
kernel_size = 3
height = 512
width = 512
stride = 2
padding = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]