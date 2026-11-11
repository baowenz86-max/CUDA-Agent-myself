import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.softmax(x, dim=2)
        x = F.elu(x)
        return x


def get_inputs():
    return [torch.randn(128, 2048, 768)]


def get_init_inputs():
    return []