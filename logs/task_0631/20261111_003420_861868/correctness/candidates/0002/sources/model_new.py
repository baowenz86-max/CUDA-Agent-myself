import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.cuda()
        x = cuda_extension.sum_dim1_kernel(x)
        x = cuda_extension.log10_kernel(x)
        return x

    def debug_forward(self, x):
        x = x.cuda()
        x = cuda_extension.sum_dim1_kernel(x)
        yield ('step_001', x)
        x = cuda_extension.log10_kernel(x)
        yield ('step_002', x)
        yield ('step_003', x)
