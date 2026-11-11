import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = F.softmax(x, dim=1)
        return x


batch_size = 512
channels = 64
height = 40
width = 40


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []