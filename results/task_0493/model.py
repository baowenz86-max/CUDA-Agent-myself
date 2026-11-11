import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, hard_shrink_lambd):
        super().__init__()
        self.hard_shrink = F.hardshrink
        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.hard_shrink_lambd = hard_shrink_lambd

    def forward(self, x):
        x = self.hard_shrink(x, lambd=self.hard_shrink_lambd)
        x = self.max_pool(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 512, 1024  # Increased width to 1024 to increase computation
kernel_size = (2, 2)
stride = 2
padding = (1, 1)
hard_shrink_lambd = 0.5


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, hard_shrink_lambd]