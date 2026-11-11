import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, x):
        x = torch.abs(x)
        x = torch.unsqueeze(x, dim=self.expand_dim)
        return x


batch_size = 512
width = 1024
height = 1024
input_size = (batch_size, height, width)
expand_dim = 1


def get_inputs():
    x = torch.randn(*input_size)
    return [x]


def get_init_inputs():
    return [input_size, expand_dim]