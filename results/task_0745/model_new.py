import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, mat1_shape, mat2_shape, bias_shape):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.mat1 = nn.Parameter(torch.randn(mat1_shape))
        self.mat2 = nn.Parameter(torch.randn(mat2_shape))

    def forward(self, x):
        x = cuda_extension.custom_mean(x, dim=1, keepdim=False)
        x = cuda_extension.custom_addmm(self.bias, x, self.mat2)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_mean(x, dim=1, keepdim=False)
        yield 'step_001', x
        x = cuda_extension.custom_addmm(self.bias, x, self.mat2)
        yield 'step_002', x
        yield 'step_003', x
