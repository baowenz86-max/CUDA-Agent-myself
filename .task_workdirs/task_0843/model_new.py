import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x, y):
        eq_result = cuda_extension.equal_forward(x, y)
        argsorted_x = cuda_extension.argsort_lastdim_forward(x)
        return eq_result, argsorted_x

    def debug_forward(self, x, y):
        eq_result = cuda_extension.equal_forward(x, y)
        yield 'step_001', eq_result
        argsorted_x = cuda_extension.argsort_lastdim_forward(x)
        yield 'step_002', argsorted_x
        yield 'step_003', (eq_result, argsorted_x)
