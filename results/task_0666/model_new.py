import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, tril_indices):
        super().__init__()
        self.tril_indices = tril_indices

    def forward(self, x):
        x = cuda_extension.triu_forward(x, self.tril_indices)
        x = cuda_extension.sum_dim23_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.triu_forward(x, self.tril_indices)
        yield 'step_001', x
        x = cuda_extension.sum_dim23_forward(x)
        yield 'step_002', x
        yield 'step_003', x
