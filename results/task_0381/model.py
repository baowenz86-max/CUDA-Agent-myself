import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardtanh = F.hardtanh
        self.instance_norm = nn.InstanceNorm3d(1)  # Assuming input channels is 1

    def forward(self, x):
        x = self.hardtanh(x)
        x = x[..., 0]  # Slicing to remove the added dimensions, assuming input size is [B, C, D, H, W]
        x = self.instance_norm(x)
        return x


# Hyperparameters for input shape
batch_size = 1
channels = 1
depth = 512  # Adjusted to be at least 512
height, width = 90, 90
extra_dim = 1  # Added to make input 6D so slicing results in 5D


def get_inputs():
    return [torch.randn(batch_size, channels, depth, height, width, extra_dim)]


# No initialization parameters needed
def get_init_inputs():
    return []