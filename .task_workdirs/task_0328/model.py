import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, pool_kernel_size, pool_stride):
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        x = F.relu(x)
        x = self.maxpool(x)
        return x


batch_size = 16
in_channels = 256  # Increased from 3 to 256 to increase computational load
width = 512
height = 224
kernel_size = (3, 3)
pool_kernel_size = (2, 2)
pool_stride = (2, 2)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, pool_kernel_size, pool_stride]