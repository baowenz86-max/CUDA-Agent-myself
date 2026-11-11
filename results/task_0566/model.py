import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = torch.abs(x)
        x = torch.gt(x, torch.tensor(0.5))
        return x


def get_inputs():
    return [torch.randn(128, 3, 1024, 1024)]


def get_init_inputs():
    return [get_inputs()[0].size()]