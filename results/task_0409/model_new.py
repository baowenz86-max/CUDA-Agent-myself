import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scalar):
        super(ModelNew, self).__init__()
        self.scalar = nn.Parameter(torch.tensor(scalar))

    def forward(self, x):
        max_val = cuda_extension.custom_max_dim1(x)
        out = cuda_extension.custom_scalar_mul(max_val, self.scalar)
        return out

    def debug_forward(self, x):
        max_val = cuda_extension.custom_max_dim1(x)
        argmax_indices = cuda_extension.custom_argmax_dim1(x)
        max_tuple = torch.return_types.max((max_val, argmax_indices))
        yield 'step_001', max_tuple
        yield 'step_002', self.scalar
        out = cuda_extension.custom_scalar_mul(max_val, self.scalar)
        yield 'step_003', out
