import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, num_groups, num_features):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.tanh = torch.tanh

    def forward(self, x):
        x = self.group_norm(x)
        x = self.tanh(x)
        return x


batch_size = 16
in_channels = 3
height, width = 2048, 512  # Increased spatial dimensions to increase computation
num_groups = 3


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, num_groups, in_channels]