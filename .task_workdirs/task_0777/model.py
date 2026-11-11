import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.mean(x, dim=self.dim)
        x = torch.softmax(x, dim=self.dim)
        return x


def get_inputs():
    return [torch.randn(512, 32, 64, 1024)]


def get_init_inputs():
    return [2]