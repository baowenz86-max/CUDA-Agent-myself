import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, eps_val=1e-05, momentum_val=0.1):
        super().__init__()
        self.arange_tensor = torch.arange(num_channels)
        self.batchnorm = nn.BatchNorm2d(num_channels, eps=eps_val, momentum=momentum_val)

    def forward(self, x):
        x = self.batchnorm(x)
        return x


batch_size = 16
num_channels = 3
height = 512
width = 512  # Increased from 256 to 512 to increase computation


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_channels]