import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.gelu_forward(x)
        x = cuda_extension.prod_dim1_keepdim_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.gelu_forward(x)
        yield 'step_001', x
        x = cuda_extension.prod_dim1_keepdim_forward(x)
        yield 'step_002', x
        yield 'step_003', x
