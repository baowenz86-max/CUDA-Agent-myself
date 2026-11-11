import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = nn.LogSoftmax()(x)
        return x


batch_size = 512
channels = 5
height = 224
width = 224


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return []  # No parameters needed for this case