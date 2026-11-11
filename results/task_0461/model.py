import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.pad = nn.ZeroPad2d(pad_size)

    def forward(self, x):
        x = self.pad(x)
        return torch.max(x, dim=1)[0]


batch_size = 512
height, width = 256, 256  # Increased from 64x64 to increase computation
pad_size = (2, 3, 2, 3)  # pad_size for nn.ZeroPad2d in (left, right, top, bottom) format


def get_inputs():
    return [torch.randn(batch_size, 32, height, width)]  # Increased channels from 1 to 32


def get_init_inputs():
    return [pad_size]