import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = nn.Hardswish()(x)
        return x


batch_size = 512  # Adjusted to ensure at least one dimension >=512
height, width = 128, 128
channels = 64  # Increased channels to make softmax more computationally intensive


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []  # No parameters needed for this combination