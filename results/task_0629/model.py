import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, multiplier_shape):
        super().__init__()
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, *inputs):
        concatenated = torch.cat(inputs, dim=1)
        multiplied = torch.mul(concatenated, self.multiplier)
        return multiplied


batch_size = 16
channels = 8
height = 384  # Increased from 128
width = 1536  # Increased from 512


def get_inputs():
    input1 = torch.randn(batch_size, channels, height, width)
    input2 = torch.randn(batch_size, channels, height, width)
    return [input1, input2]


def get_init_inputs():
    return [2, 16, (1,)]