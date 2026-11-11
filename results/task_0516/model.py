import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_channels, num_groups, num_channels_per_group):
        super().__init__()
        self.group_norm = F.group_norm
        self.num_groups = num_groups

    def forward(self, x):
        x = self.group_norm(x, self.num_groups)
        return x


batch_size = 512
height, width = 320, 320  # Increased spatial dimensions to increase computation
num_channels = 3
num_groups = 3


def get_inputs():
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    return [num_channels, num_groups, num_channels // num_groups]