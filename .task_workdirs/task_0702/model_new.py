import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        x = cuda_extension.sum_dim1(x)
        x = cuda_extension.log_softmax_dim0(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.sum_dim1(x)
        yield 'step_001', x
        x = cuda_extension.log_softmax_dim0(x)
        yield 'step_002', x
        yield 'step_003', x
