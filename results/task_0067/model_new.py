import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = cuda_extension.elu_forward(x)
        x = cuda_extension.mean_dim1_keepdim_forward(x)
        x = cuda_extension.add_scalar_forward(x, self.mean)
        return x

    def debug_forward(self, x):
        x = cuda_extension.elu_forward(x)
        yield 'step_001', x
        x = cuda_extension.mean_dim1_keepdim_forward(x)
        yield 'step_002', x
        x = cuda_extension.add_scalar_forward(x, self.mean)
        yield 'step_003', x
        yield 'step_004', x
