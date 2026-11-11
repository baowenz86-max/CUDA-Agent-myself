import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.gelu(x)
        return x


batch_size = 16
height, width = 512, 256  # Increased input dimensions to increase computation


def get_inputs():
    x = torch.randn(batch_size, 256, height, width)  # Increased channels from 1 to 256
    return [x]


def get_init_inputs():
    return []