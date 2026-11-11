import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        x = cuda_extension.erf_forward(x)
        x = cuda_extension.erf_forward(x)
        x = cuda_extension.median_dim0_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.erf_forward(x)
        yield 'step_001', x
        x = cuda_extension.erf_forward(x)
        yield 'step_002', x
        x = cuda_extension.median_dim0_forward(x)
        yield 'step_003', x
        yield 'step_004', x
