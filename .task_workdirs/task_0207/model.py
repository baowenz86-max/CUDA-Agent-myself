import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, max_channels):
        super().__init__()
        self.max_channels = max_channels

    def forward(self, x):
        x = torch.clamp(x, min=0.0, max=1.0)
        x = F.softmax(x, dim=1)
        return x[:, : self.max_channels, :, :]


batch_size = 512  # Changed from 16 to 512 to make batch dimension >=512
height = 128  # Increased from 32 to 128 to increase computational load
width = 128   # Increased from 32 to 128 to increase computational load
max_channels = 16


def get_inputs():
    x = torch.randn(batch_size, max_channels * 2, height, width)
    return [x]


def get_init_inputs():
    return [max_channels]