import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.relu(x)
        return x


batch_size = 512
height, width = 1024, 1024


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []