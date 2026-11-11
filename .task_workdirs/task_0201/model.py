import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_channels, num_groups, eps, affine):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def forward(self, x):
        x = self.group_norm(x)
        x = F.gelu(x)
        return x


num_channels = 64
num_groups = 2
eps = 1e-5
affine = True


def get_init_inputs():
    return [num_channels, num_groups, eps, affine]


def get_inputs():
    batch_size = 512
    height = 64  # Increased spatial dimension to increase computation
    width = 64   # Increased spatial dimension to increase computation
    return [torch.randn(batch_size, num_channels, height, width)]