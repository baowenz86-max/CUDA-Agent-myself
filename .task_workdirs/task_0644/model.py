import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.floor(x)
        return x


batch_size = 1024
width = 512
height = 512


def get_inputs():
    return [torch.randn(batch_size, 1, height, width)]


def get_init_inputs():
    return []