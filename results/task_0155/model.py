import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x):
        x = self.conv3d(x)
        x = torch.tanh(x)
        return x


batch_size = 512
in_channels = 3
out_channels = 64
depth, height, width = 16, 16, 16  # Increased spatial dimensions
kernel_size = (3, 3, 3)


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]