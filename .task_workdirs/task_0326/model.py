import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.hardsigmoid(x)
        x = torch.mean(x, dim=(2, 3))  # Mean over last two dimensions (batch and channel wise mean)
        return x


batch_size = 128
channels = 64
height, width = 1024, 64  # Increased height and width to increase runtime


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []  # No init inputs needed in this case