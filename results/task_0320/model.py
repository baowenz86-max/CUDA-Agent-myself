import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.hardtanh(x)
        x = F.leaky_relu(x)
        return x


batch_size = 16
channels = 3
height = 4096
width = 2048


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []