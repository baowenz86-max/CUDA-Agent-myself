import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.sum(x, dim=self.dim)
        x = torch.softmax(x, dim=self.dim)
        return x


batch_size = 64
channels = 3
height = 1024
width = 4096
dim = 0


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [dim]