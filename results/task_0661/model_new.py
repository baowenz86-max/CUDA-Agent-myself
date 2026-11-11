import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size

    def forward(self, x):
        mean = cuda_extension.mean_dim(x, self.input_size)
        equal = cuda_extension.equal_expand_as(x, mean)
        return mean, equal

    def debug_forward(self, x):
        mean = cuda_extension.mean_dim(x, self.input_size)
        yield 'step_001', mean
        equal = cuda_extension.equal_expand_as(x, mean)
        yield 'step_002', equal
        yield 'step_003', (mean, equal)
