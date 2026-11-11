import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = torch.sum(x, dim=1, keepdim=True)
        x = self.leaky_relu(x)
        return x


batch_size = 512  # Increased to 512 to increase computation
width = 1024     # Increased to 1024 to increase computation
height = 256     # Increased to 256 to increase computation
in_channels = 3
out_channels = 64


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels]