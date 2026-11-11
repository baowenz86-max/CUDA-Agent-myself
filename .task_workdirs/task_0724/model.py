import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = nn.GELU()(x)
        return x


batch_size = 512
height, width = 512, 512


def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]


def get_init_inputs():
    return []