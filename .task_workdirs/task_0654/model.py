import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, reduction_indices):
        super().__init__()
        self.input_size = input_size
        self.reduction_indices = reduction_indices

    def forward(self, x):
        x = torch.abs(x)
        x = torch.argmin(x, dim=self.reduction_indices)
        return x


batch_size = 16
input_size = (1024, 512, 128)  # Increased first dimension to increase computation
reduction_indices = 1


def get_inputs():
    return [torch.randn(batch_size, *input_size)]


def get_init_inputs():
    return [input_size, reduction_indices]