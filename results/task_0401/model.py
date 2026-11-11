import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x, _ = torch.max(x, dim=1, keepdim=True)
        x = self.maxpool(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 512, 1024  # Increased width to 1024 to increase computation
kernel_size = 3
stride = 2
padding = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]