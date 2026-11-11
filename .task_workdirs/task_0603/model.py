import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.gelu(x)
        x = torch.mean(x, dim=1)
        return x


batch_size = 512
d1, d2, d3 = 64, 64, 128


def get_inputs():
    return [torch.randn(batch_size, d1, d2, d3)]


def get_init_inputs():
    return []