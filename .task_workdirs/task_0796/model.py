import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        x = torch.tan(x)
        return x


batch_size = 128
height, width = 2048, 2048  # Increased spatial dimensions to increase computation
channels = 3


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []