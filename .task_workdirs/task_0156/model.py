import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.pad1d = nn.ConstantPad1d(padding, 0)

    def forward(self, x):
        x = self.unfold(x)
        x = self.pad1d(x)
        return x


batch_size = 24  # Increased from 16 to 24 to increase runtime
in_channels = 3
height = 512
width = 256
kernel_size = 3
stride = 1
padding = (1, 2)
dilation = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]