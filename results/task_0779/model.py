import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim, input_shape1, input_shape2):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=self.dim)
        x = torch.prod(x, dim=self.dim)
        return x


dim = 1
batch_size = 512  # Increased to ensure at least one dimension >=512
channels1 = 32
channels2 = 16
height, width = 256, 256  # Increased spatial dimensions to increase computation


def get_inputs():
    x1 = torch.randn(batch_size, channels1, height, width)
    x2 = torch.randn(batch_size, channels2, height, width)
    return [x1, x2]


def get_init_inputs():
    return [dim, (batch_size, channels1, height, width), (batch_size, channels2, height, width)]