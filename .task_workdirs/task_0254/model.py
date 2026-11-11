import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size, min_val, max_val):
        super().__init__()
        self.input_size = input_size
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val)


batch_size = 16
in_channels = 3
height = 4096
width = 4096


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [16, -1, 1]