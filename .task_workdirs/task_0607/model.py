import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, channels, kernel_size, dilation, input_shape):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation)
        self.channels = channels
        self.input_shape = input_shape

    def forward(self, x):
        x = self.unfold(x)
        x = torch.abs(x)
        return x


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
width = 640  # Increased from 512 to 640
height = 320  # Increased from 256 to 320
dilation = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, dilation, (height, width)]