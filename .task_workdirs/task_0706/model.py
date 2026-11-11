import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = torch.sum(x, dim=1, keepdim=True)  # assuming the input is in BCHW format
        x = self.pool(x)
        return x


batch_size = 16
in_channels = 3
out_channels = 1
height = 2560  # Increased to increase computation
width = 2560   # Increased to increase computation
kernel_size = 2
stride = 2
padding = 0


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]