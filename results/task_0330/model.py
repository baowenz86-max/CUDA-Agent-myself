import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, bias_shape):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, mat1, mat2):
        return torch.baddbmm(self.bias, mat1, mat2)


batch_size = 512  # Increased to ensure at least one dimension >=512
input_dim = 64
output_dim = 32
bias_shape = (batch_size, output_dim, input_dim)


def get_inputs():
    mat1 = torch.randn(batch_size, output_dim, 8192)  # Increased middle dimension to 8192
    mat2 = torch.randn(batch_size, 8192, input_dim)   # Increased middle dimension to 8192
    return [mat1, mat2]


def get_init_inputs():
    return [batch_size, input_dim, output_dim, bias_shape]