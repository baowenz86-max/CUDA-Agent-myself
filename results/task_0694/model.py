import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_axes):
        super().__init__()
        self.num_axes = num_axes

    def forward(self, x):
        x = torch.sum(x, dim=self.num_axes)
        x = nn.Sigmoid()(x)
        return x


batch_size = 512
height, width = 512, 512  # Increased from 64x64 to 512x512 to increase computation
channels = 3
num_axes = 1


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [num_axes]