import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super().__init__()
        self.maxpool1d = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x):
        x = self.maxpool1d(x)
        x = self.conv1d(x)
        return x


batch_size = 512
in_channels = 16
out_channels = 32
height = 256
width = 8192  # Increased from 128 to 8192
kernel_size = 3
stride = 2
padding = 1
dilation = 1
bias = True


def get_inputs():
    return [torch.randn(batch_size, in_channels, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, bias]