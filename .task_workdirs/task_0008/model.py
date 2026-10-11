import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias_shape):
        super().__init__()
        self.pad = nn.ConstantPad2d(padding, padding[0])
        self.rotate = lambda x: torch.rot90(x, k=1, dims=(2, 3))
        self.is_inf = torch.isinf
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.pad(x)
        x = self.rotate(x)
        x = self.conv(x)
        mask = self.is_inf(x)
        x = x.masked_fill(mask, 0.0)
        x = x + self.bias
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 128, 128  # Increased spatial dimensions to increase computation
kernel_size = 3
padding = (1, 1)
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding, bias_shape]