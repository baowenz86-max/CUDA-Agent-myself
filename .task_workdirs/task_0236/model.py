import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.softmin(x, dim=1)


batch_size = 512
height, width = 512, 512


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []