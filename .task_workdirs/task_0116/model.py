import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, multiply_factor, kernel_size):
        super().__init__()
        self.multiply_factor = multiply_factor
        self.avgpool = nn.AvgPool1d(kernel_size)

    def forward(self, x):
        x = torch.multiply(x, self.multiply_factor)
        x = self.avgpool(x)
        return x


def get_inputs():
    batch_size = 16
    input_size = 64
    sequence_length = 100000  # Increased to increase computation
    x = torch.randn(batch_size, input_size, sequence_length)
    return [x]


def get_init_inputs():
    input_size = 64
    multiply_factor = torch.tensor([2.0])
    kernel_size = 3
    return [input_size, multiply_factor, kernel_size]