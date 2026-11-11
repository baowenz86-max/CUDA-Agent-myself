import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, reduction_ratio):
        super().__init__()
        self.identity = nn.Identity()
        self.reduction_ratio = reduction_ratio

    def forward(self, x):
        # Apply max operation
        x, _ = torch.max(x, dim=1, keepdim=True)
        # Apply identity operation
        x = self.identity(x)
        x = x / self.reduction_ratio
        return x


batch_size = 512
num_channels = 3
height, width = 320, 320  # Increased from 64x64 to 320x320
reduction_ratio = 2.0


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_channels, reduction_ratio]