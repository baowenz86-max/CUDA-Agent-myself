import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.clamp(x, min=0.0, max=1.0)
        x = F.hardswish(x)
        return x


batch_size = 512


def get_inputs():
    return [torch.randn(batch_size, 64, 128, 128)]


def get_init_inputs():
    return []