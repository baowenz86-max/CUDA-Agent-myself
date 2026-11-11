import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, clamp_min, clamp_max, argmin_dim):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.argmin_dim = argmin_dim

    def forward(self, x):
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        x = torch.argmin(x, dim=self.argmin_dim)
        return x


def get_inputs():
    batch_size = 16
    channels = 3
    height = 2048  # Increased from 512 to 2048 to increase computation
    width = 2048   # Increased from 256 to 2048 to increase computation
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [0.0, 1.0, 1]