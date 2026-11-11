import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = F.gelu(x)
        return x


batch_size = 1024  # Increased from 16 to increase computational load
channels = 3
height = 512
width = 256


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []