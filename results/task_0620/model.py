import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super().__init__()
        self.hardtanh = nn.Hardtanh()
        self.avgpool3d = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.hardtanh(x)
        x = self.avgpool3d(x)
        return x


batch_size = 8  # Increased batch size to increase computation
in_channels = 3
height, width, depth = 512, 64, 64
kernel_size = (2, 2, 2)
stride = (1, 1, 1)
padding = (0, 0, 0)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width, depth)]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]