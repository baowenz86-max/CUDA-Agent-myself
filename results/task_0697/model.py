import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, leaky_relu_slope):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.leaky_relu(x)
        return x


batch_size = 512
height, width = 64, 64
num_channels = 1024  # Increased from 3 to increase computation
leaky_relu_slope = 0.01


def get_inputs():
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    return [num_channels, leaky_relu_slope]