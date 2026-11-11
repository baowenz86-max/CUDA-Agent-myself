import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)

    def forward(self, x):
        x = F.leaky_relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2d(x)
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 256, 256
kernel_size = (3, 3)
stride = 1
dilation = 1
bias = True


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation, bias]