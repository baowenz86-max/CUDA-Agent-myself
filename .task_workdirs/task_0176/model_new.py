import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.log_softmax_sum_forward(x)
        return x

    def debug_forward(self, x):
        x_log_softmax = cuda_extension.log_softmax_forward(x)
        yield ('step_001', x_log_softmax)
        x_sum = cuda_extension.sum_dim_forward(x_log_softmax, dim=1)
        yield ('step_002', x_sum)
        yield ('step_003', x_sum)
