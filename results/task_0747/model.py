import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, reduction_dim, transpose_dim):
        super().__init__()
        self.reduction_dim = reduction_dim
        self.transpose_dim = transpose_dim

    def forward(self, x):
        x = torch.mean(x, dim=self.reduction_dim)
        x = torch.transpose(x, self.transpose_dim[0], self.transpose_dim[1])
        return x


batch_size = 16
channels = 3
height = 8192
width = 2048
reduction_dim = 1
transpose_dim = (1, 2)


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [reduction_dim, transpose_dim]