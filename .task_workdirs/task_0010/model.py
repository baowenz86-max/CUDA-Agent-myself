import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv2d(x)
        x = torch.triu(x)
        x = torch.bitwise_xor(x.type(torch.uint8), self.bias.type(torch.uint8).expand_as(x))
        return x.float()


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 128, 128  # Increased from 32x32 to 128x128 to increase computation
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]