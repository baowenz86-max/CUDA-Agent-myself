import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, batch_size, input_dim, mat1_dim, mat2_dim):
        super().__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.mat1_dim = mat1_dim
        self.mat2_dim = mat2_dim

    def forward(self, input, mat1, mat2):
        baddbmm_result = torch.baddbmm(input, mat1, mat2)
        equal_result = torch.equal(baddbmm_result, input)
        return equal_result, baddbmm_result


batch_size = 512
input_dim = 256  # Increased from 64 to 256
mat1_dim = 512   # Increased from 128 to 512
mat2_dim = 256   # Increased from 64 to 256


def get_inputs():
    input = torch.randn(batch_size, mat1_dim, mat2_dim)
    mat1 = torch.randn(batch_size, mat1_dim, input_dim)
    mat2 = torch.randn(batch_size, input_dim, mat2_dim)
    return [input, mat1, mat2]


def get_init_inputs():
    return [batch_size, input_dim, mat1_dim, mat2_dim]