import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=3)

    def forward(self, x):
        return self.group_norm(x)


batch_size = 512
num_groups = 3
channels = 3
height = 384
width = 384


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [num_groups]