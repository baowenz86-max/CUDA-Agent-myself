import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.abs_forward(x)
        x = cuda_extension.log_softmax_forward(x, dim=1)
        return x

    def debug_forward(self, x):
        x = cuda_extension.abs_forward(x)
        yield 'step_001', x
        x = cuda_extension.log_softmax_forward(x, dim=1)
        yield 'step_002', x
        yield 'step_003', x
