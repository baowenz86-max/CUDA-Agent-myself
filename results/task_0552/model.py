import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor

    def forward(self, x):
        x = F.elu(x)
        x = torch.abs(x)
        return x


reduction_factor = 0.5
batch_size = 512
channels = 16
height = 320
width = 320


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [reduction_factor]