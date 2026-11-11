import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, num_groups):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, x):
        # Compute the maximum value over the last dimension of the input tensor
        max_value, _ = torch.max(x, dim=-1, keepdim=True)

        # Apply group normalization to the input tensor
        x = self.group_norm(x)

        # Add the maximum value to the normalized tensor
        x = x + max_value
        return x


batch_size = 512
num_channels = 3
height, width = 256, 256  # Increased from 64x64 to 256x256
num_groups = 3


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_channels, num_groups]