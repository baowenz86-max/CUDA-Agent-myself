import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = x * self.out_channels
        return x


batch_size = 512
height, width = 512, 512
in_channels = 3
out_channels = 4


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels]