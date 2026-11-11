import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.softplus(x)
        x = F.leaky_relu(x)
        return x


batch_size = 512
channels = 100
height = 64
width = 64


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []