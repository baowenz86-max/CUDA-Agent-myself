import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scalar):
        super(ModelNew, self).__init__()
        self.scalar = nn.Parameter(torch.tensor(scalar))

    def forward(self, x):
        max_val = cuda_extension.custom_max_dim1(x)
        scalar_tensor = torch.scalar_tensor(self.scalar.item())
        out = cuda_extension.custom_scalar_mul(max_val, scalar_tensor)
        return out

    def debug_forward(self, x):
        max_val = cuda_extension.custom_max_dim1(x)
        argmax_indices = cuda_extension.custom_argmax_dim1(x)
        max_tuple = torch.return_types.max((max_val, argmax_indices))
        yield 'step_001', max_tuple
        scalar_tensor = torch.scalar_tensor(self.scalar.item())
        yield 'step_002', scalar_tensor
        out = cuda_extension.custom_scalar_mul(max_val, scalar_tensor)
        yield 'step_003', out
