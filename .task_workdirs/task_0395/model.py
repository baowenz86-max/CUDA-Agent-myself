import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, ceil_mode=False):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    def forward(self, x):
        x = self.avg_pool(x)
        x = F.leaky_relu(x)  # Using leaky_relu function from nn.functional
        return x


batch_size = 80  # Increased batch size to increase runtime
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
height = 512
width = 256
stride = 1
padding = (1, 2)
ceil_mode = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, ceil_mode]