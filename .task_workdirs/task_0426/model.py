import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, flip_dims):
        super().__init__()
        self.flip_dims = flip_dims

    def forward(self, x):
        x = torch.abs(x)
        x = torch.flip(x, self.flip_dims)
        return x


def get_inputs():
    return [torch.randn(512, 16, 256, 256)]  # Increased spatial dimensions to 256x256


def get_init_inputs():
    return [tuple(range(2, 4))]  # Flipping on the last two dimensions