import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, ndim, axes, k):
        super().__init__()
        self.ndim = ndim
        self.axes = axes
        self.k = k

    def forward(self, x):
        x = F.softmax(x, dim=self.ndim)
        x = torch.rot90(x, self.k, self.axes)
        return x


batch_size, channels, height, width = 16, 256, 512, 256  # Increased channels from 3 to 256
ndim = 1
axes = (2, 3)
k = 1


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [ndim, axes, k]