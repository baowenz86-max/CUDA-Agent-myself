import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_list):
        x = cuda_extension.stack_tensors(x_list, dim=0)
        x = cuda_extension.softmax_dim0(x)
        return x

    def debug_forward(self, x_list):
        x = cuda_extension.stack_tensors(x_list, dim=0)
        yield 'step_001', x
        x = cuda_extension.softmax_dim0(x)
        yield 'step_002', x
        yield 'step_003', x
