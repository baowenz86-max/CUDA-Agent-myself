import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape, index_shape, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, index):
        mask = torch.isnan(x)
        x_gathered = torch.gather(x, self.dim, index)
        return mask, x_gathered


batch_size = 512
input_shape = (batch_size, 3, 512, 512)  # Increased spatial dimensions to 512x512
index_shape = (batch_size, 1, 512, 512)  # Matched spatial dimensions with input
dim = 1


def get_inputs():
    x = torch.randn(input_shape)
    index = torch.randint(0, input_shape[dim], index_shape)
    return [x, index]


def get_init_inputs():
    return [input_shape, index_shape, dim]