import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation
        )

    def forward(self, x):
        x = torch.mean(x, dim=0, keepdim=True)
        return self.conv3d(x)


def get_inputs():
    batch_size = 512  # Adjusted to ensure at least one dimension >=512
    in_channels = 3
    depth, height, width = 128, 128, 128  # Increased spatial dimensions
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    in_channels = 3
    out_channels = 8
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]