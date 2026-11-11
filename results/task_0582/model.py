import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, eps):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_channels, num_channels)
        self.eps = eps

    def forward(self, x):
        x = self.group_norm(x)
        x = torch.log_softmax(x, dim=1)
        return x


num_channels = 16
eps = 1e-5
height, width = 128, 128  # Increased spatial dimensions to increase computation
batch_size = 512


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_channels, eps]