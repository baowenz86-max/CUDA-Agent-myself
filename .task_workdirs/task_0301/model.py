import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


batch_size = 2
matrix_size = 2048


def get_inputs():
    x = torch.randn(batch_size, matrix_size, matrix_size)
    y = torch.randn(batch_size, matrix_size, matrix_size)
    return [x, y]


def get_init_inputs():
    return []