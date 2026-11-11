import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, eps):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

    def forward(self, x):
        x = cuda_extension.instance_norm2d_forward(x, self.num_channels, self.eps)
        x = cuda_extension.log_softmax_dim1_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.instance_norm2d_forward(x, self.num_channels, self.eps)
        yield 'step_001', x
        x = cuda_extension.log_softmax_dim1_forward(x)
        yield 'step_002', x
        yield 'step_003', x
