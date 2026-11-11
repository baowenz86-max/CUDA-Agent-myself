import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, kernel_size, dim):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size)
        self.dim = dim

    def forward(self, x):
        x = self.unfold(x)
        x = torch.sum(x, dim=self.dim)
        return x


batch_size = 16
in_channels = 3
height, width = 512, 512  # Increased width to 512 to increase computational load
kernel_size = (3, 5)
dim = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, dim]