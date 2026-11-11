import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log_softmax(x, dim=1)
        x = F.hardshrink(x)
        return x


batch_size, channels, height, width = 1024, 3, 224, 224


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []