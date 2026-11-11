import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.tanh(x)
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 288, 288  # Increased spatial dimensions to increase runtime
kernel_size = 3
stride = 2
padding = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]