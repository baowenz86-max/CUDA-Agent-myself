import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_val, _ = cuda_extension.max_dim_keepdim_forward(x, 1, True)
        cloned = cuda_extension.clone_forward(x)
        return max_val, cloned

    def debug_forward(self, x):
        max_val, _ = cuda_extension.max_dim_keepdim_forward(x, 1, True)
        yield 'step_001', (max_val, _)
        cloned = cuda_extension.clone_forward(x)
        yield 'step_002', cloned
        yield 'step_003', (max_val, cloned)
