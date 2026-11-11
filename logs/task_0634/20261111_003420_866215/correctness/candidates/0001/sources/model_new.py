import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, inplace):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        x = cuda_extension.to_cuda(x)
        mean = cuda_extension.mean_reduce_keepdim(x, dim=[1, 2, 3])
        x = cuda_extension.where_less_zero(x, mean)
        return x

    def debug_forward(self, x):
        x = cuda_extension.to_cuda(x)
        mean = cuda_extension.mean_reduce_keepdim(x, dim=[1, 2, 3])
        yield 'step_001', mean
        x = cuda_extension.where_less_zero(x, mean)
        yield 'step_002', x
        yield 'step_003', x
