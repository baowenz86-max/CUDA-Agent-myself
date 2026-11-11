import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape, src_shape, dim1, dim2, offset):
        super().__init__()
        self.src = nn.Parameter(torch.randn(src_shape))
        self.dim1 = dim1
        self.dim2 = dim2
        self.offset = offset

    def forward(self, x):
        x = torch.diagonal_scatter(x, self.src, offset=self.offset, dim1=self.dim1, dim2=self.dim2)
        return x


batch_size = 512
input_shape = (batch_size, 1024, 1024)  # Increased spatial dimensions to 1024x1024
src_shape = (batch_size, 1024)  # Matched src length to spatial dimension size
dim1 = 1
dim2 = 2
offset = 0


def get_inputs():
    x = torch.randn(input_shape)
    return [x]


def get_init_inputs():
    return [input_shape, src_shape, dim1, dim2, offset]