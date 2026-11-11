import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, bias_shape):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, mat1, mat2):
        return cuda_extension.baddbmm_forward(self.bias, mat1, mat2, 1.0, 1.0)

    def debug_forward(self, mat1, mat2):
        out = cuda_extension.baddbmm_forward(self.bias, mat1, mat2, 1.0, 1.0)
        yield 'step_001', out
