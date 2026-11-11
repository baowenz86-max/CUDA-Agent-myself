import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pooling_kernel):
        super().__init__()
        self.pooling_kernel = pooling_kernel

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.custom_adaptive_avg_pool2d(x, output_size=5)
        return x

    def debug_forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.custom_adaptive_avg_pool2d(x, output_size=5)
        yield 'step_001', x
        yield 'step_002', x
