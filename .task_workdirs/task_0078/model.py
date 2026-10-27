import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, num_groups, einsum_eqn, transpose_dims):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups, num_channels)
        self.einsum_eqn = einsum_eqn
        self.transpose_dims = transpose_dims

    def forward(self, x):
        x = self.groupnorm(x)
        x = torch.einsum(self.einsum_eqn, x)
        x = torch.transpose(x, self.transpose_dims[0], self.transpose_dims[1])
        return x


def get_inputs():
    batch_size = 16
    num_channels = 3
    height = 1024  # Increased from 512 to 1024
    width = 1024  # Increased from 512 to 1024
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    num_channels = 3
    num_groups = 3
    einsum_eqn = "bhwc->bchw"
    transpose_dims = (1, 2)
    return [num_channels, num_groups, einsum_eqn, transpose_dims]