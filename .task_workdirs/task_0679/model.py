import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        sum_result = torch.sum(x, dim=self.dim)
        argmax_result = torch.argmax(sum_result)
        return argmax_result


batch_size, channels, height, width = 16, 3, 4096, 4096
dim = 1


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [dim]