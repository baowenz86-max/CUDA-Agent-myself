import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape, src_shape, dim1, dim2, offset):
        super().__init__()
        self.src = nn.Parameter(torch.randn(src_shape))
        self.dim1 = dim1
        self.dim2 = dim2
        self.offset = offset

    def forward(self, x):
        x = cuda_extension.diagonal_scatter(x, self.src, self.offset, self.dim1, self.dim2)
        return x

    def debug_forward(self, x):
        x = cuda_extension.diagonal_scatter(x, self.src, self.offset, self.dim1, self.dim2)
        yield "step_001", x
        yield "step_002", x
