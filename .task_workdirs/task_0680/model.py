import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=self.dim)
        x = torch.nansum(x, dim=self.dim)
        return x


batch_size = 32
channels = 3
height, width = 1024, 1024  # Increased spatial dimensions


def get_inputs():
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    return [x1, x2]


def get_init_inputs():
    return [1]  # dimension to concatenate along