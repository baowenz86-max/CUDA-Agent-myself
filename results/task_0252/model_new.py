import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.softmax_dim2(x)
        x = cuda_extension.elu(x)
        return x

    def debug_forward(self, x):
        x_softmax = cuda_extension.softmax_dim2(x)
        yield ('step_001', x_softmax)
        x_elu = cuda_extension.elu(x_softmax)
        yield ('step_002', x_elu)
        yield ('step_003', x_elu)
