import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super().__init__()
        self.maxpool3d = nn.MaxPool3d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.maxpool3d(x)
        x = torch.prod(x, dim=1, keepdim=True)
        return x


batch_size = 512
in_channels = 3
height, width, depth = 64, 64, 64  # Increased spatial dimensions from 32 to 64
kernel_size = (2, 2, 2)
stride = (2, 2, 2)
padding = (1, 1, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]