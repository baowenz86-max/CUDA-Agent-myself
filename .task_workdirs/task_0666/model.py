import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, tril_indices):
        super().__init__()
        self.tril_indices = tril_indices

    def forward(self, x):
        x = torch.triu(x, diagonal=self.tril_indices)
        x = torch.sum(x, dim=(2, 3))
        return x


batch_size = 16
input_channels = 3
height = 2048
width = 1280
tril_indices = 1


def get_inputs():
    x = torch.randn(batch_size, input_channels, height, width)
    return [x]


def get_init_inputs():
    return [input_channels, tril_indices]