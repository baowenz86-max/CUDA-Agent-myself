import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_tensor_shape, elements_shape, split_sizes):
        super().__init__()
        self.elements = nn.Parameter(torch.randn(elements_shape))

    def forward(self, x):
        mask = torch.isin(x, self.elements)
        parts = torch.split_with_sizes(x, split_sizes, dim=1)
        return mask, parts


batch_size = 512
input_channels = 10
height, width = 64, 64  # Increased from 32x32 to 64x64
elements_shape = (input_channels,)
split_sizes = [5, 3, 2]


def get_inputs():
    x = torch.randn(batch_size, input_channels, height, width)
    return [x]


def get_init_inputs():
    return [input_channels, elements_shape, split_sizes]