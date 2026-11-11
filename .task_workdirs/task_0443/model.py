import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.gelu(x)
        x = torch.cumsum(x, dim=1)
        return x


batch_size, channels, height, width = 512, 128, 64, 64


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []