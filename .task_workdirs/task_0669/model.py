import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = x + self.bias
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 256, 256  # Increased spatial dimensions to increase computation
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1)  # Fixed bias shape to match summed channel dimension


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]