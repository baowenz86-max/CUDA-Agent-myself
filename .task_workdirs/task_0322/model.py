import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_channels, groups):
        super().__init__()
        self.channels_shuffle = lambda x: F.native_channel_shuffle(x, groups)

    def forward(self, x):
        x = self.channels_shuffle(x)
        return x


batch_size = 512
num_channels = 32
height, width = 128, 128  # Increased spatial dimensions
groups = 2


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_channels, groups]