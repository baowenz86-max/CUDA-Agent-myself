import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pool_size, stride, dilation, return_indices=False):
        super().__init__()
        self.maxpool2d = nn.MaxPool2d(
            kernel_size=pool_size, stride=stride, dilation=dilation, return_indices=return_indices
        )

    def forward(self, x):
        x = self.maxpool2d(x)
        return x


batch_size = 16
in_channels = 3
height, width = 4096, 2048  # Increased spatial dimensions to increase computation
pool_size = (2, 2)
stride = 2
dilation = 1
return_indices = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [pool_size, stride, dilation, return_indices]