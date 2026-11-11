import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x, y):
        eq_result = torch.eq(x, y)
        argsorted_x = torch.argsort(x, dim=-1)
        return eq_result, argsorted_x


batch_size = 128
height, width = 512, 320  # Increased width to increase computational load
threshold = 0.5


def get_inputs():
    x = torch.randn(batch_size, height, width)
    y = torch.randn(batch_size, height, width)
    return [x, y]


def get_init_inputs():
    return [threshold]