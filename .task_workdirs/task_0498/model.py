import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.gelu(x)
        x = torch.prod(x, dim=1, keepdim=True)
        return x


batch_size = 16
channels = 128  # Increased from 3 to increase computation
height = 512
width = 256


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []