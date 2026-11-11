import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        x = torch.sum(x, dim=self.dim)
        return x


batch_size = 16
channels = 3
height, width = 4096, 2048  # Increased spatial dimensions to increase computation
dim = 2  # The dim for the sum operation


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [dim]  # The dim for the sum operation