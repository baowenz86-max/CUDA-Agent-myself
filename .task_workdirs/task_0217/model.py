import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x):
        x = torch.log_softmax(x, dim=1)
        x = x * self.multiplier
        return x


batch_size = 16
height, width = 1536, 256  # Increased height to increase log_softmax computation
multiplier = 0.5


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return [multiplier]