import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.relu(x)
        x = torch.selu(x)
        return x


# Test case:
batch_size = 512
height, width = 192, 192  # Increased from 64x64 to 192x192
channels = 16


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []