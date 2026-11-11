import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, num_channels, num_groups):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.group_norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.group_norm(x)
        return x


# Hyperparameters adjusted to ensure input has at least one dimension >=512
batch_size = 16
num_channels = 3
width = 512  # Increased width from 256 to 512 to increase runtime
height = 512
num_features = 3
num_groups = 3


def get_inputs():
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    return [num_features, num_channels, num_groups]