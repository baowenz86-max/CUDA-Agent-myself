import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = cuda_extension.custom_mean(x, self.dim)
        x = cuda_extension.custom_softmax(x, self.dim)
        return x

    def debug_forward(self, x):
        y = cuda_extension.custom_mean(x, self.dim)
        yield ('step_001', y)
        z = cuda_extension.custom_softmax(y, self.dim)
        yield ('step_002', z)
        yield ('step_003', z)
