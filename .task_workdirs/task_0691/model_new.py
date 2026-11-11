import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # Move inputs to CUDA if they are not already (reference execution device is cuda)
        if not x.is_cuda:
            x = x.cuda()
        if not y.is_cuda:
            y = y.cuda()
        # Custom operator computes MSE loss (mean squared error) and returns scalar.
        # torch.sum on scalar is identity, so no separate sum needed.
        return cuda_extension.mse_sum_forward(x, y)

    def debug_forward(self, *inputs):
        x, y = inputs
        if not x.is_cuda:
            x = x.cuda()
        if not y.is_cuda:
            y = y.cuda()
        loss = cuda_extension.mse_sum_forward(x, y)
        yield 'step_001', loss
        yield 'step_002', loss
        yield 'step_003', loss
