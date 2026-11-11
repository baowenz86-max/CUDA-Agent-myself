import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim, soft_max_dim):
        super().__init__()
        self.soft_max_dim = soft_max_dim
        self.add_value = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        x = cuda_extension.custom_softmax_forward(x, self.soft_max_dim)
        x = cuda_extension.custom_add_forward(x, self.add_value)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_softmax_forward(x, self.soft_max_dim)
        yield 'step_001', x
        x = cuda_extension.custom_add_forward(x, self.add_value)
        yield 'step_002', x
        yield 'step_003', x
