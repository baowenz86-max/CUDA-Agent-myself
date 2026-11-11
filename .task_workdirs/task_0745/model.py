import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, mat1_shape, mat2_shape, bias_shape):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.mat1 = nn.Parameter(torch.randn(mat1_shape))
        self.mat2 = nn.Parameter(torch.randn(mat2_shape))

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=False)
        x = torch.addmm(self.bias, x, self.mat2)
        return x


batch_size = 128
dim1 = 512  # Increased to make mean operation more expensive
dim2 = 512
dim3 = 262144  # Significantly increased to make matrix multiplication more expensive


def get_inputs():
    return [torch.randn(batch_size, dim1, dim2)]  # Shape: (128, 512, 512)


def get_init_inputs():
    return [(batch_size, dim2), (dim2, dim3), (batch_size, dim3)]