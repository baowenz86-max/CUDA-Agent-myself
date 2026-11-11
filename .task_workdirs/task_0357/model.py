import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, pooling_size):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride=pooling_size)

    def forward(self, x):
        x = F.elu(self.maxpool(x))
        return x


batch_size = 16
in_channels = 3
height, width = 2560, 1280  # Increased spatial dimensions to increase computation
kernel_size = (3, 5)
pooling_size = (2, 2)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, pooling_size]