import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.leaky_relu(x)
        x = F.leaky_relu(x)
        return x


batch_size = 1000  # Increased to 1000 to increase computation
width = 512
height = 512  # Increased from 256 to 512 to increase tensor size


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return []