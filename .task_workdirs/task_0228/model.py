import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, eps_value):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_channels, eps=eps_value)

    def forward(self, x):
        x = torch.triu(x)
        x = self.batch_norm(x)
        return x


batch_size = 512
num_channels = 3
height, width = 128, 128  # Increased from 32x32 to 128x128
eps_value = 1e-05


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_channels, eps_value]