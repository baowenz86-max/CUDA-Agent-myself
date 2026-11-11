import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, mean_dim):
        super().__init__()
        self.mean_dim = mean_dim

    def forward(self, x):
        mean = cuda_extension.custom_mean(x, self.mean_dim, True)
        x = cuda_extension.custom_gt(x, mean)
        return x

    def debug_forward(self, x):
        mean = cuda_extension.custom_mean(x, self.mean_dim, True)
        x = cuda_extension.custom_gt(x, mean)
        yield 'step_001', x
        yield 'step_002', x
