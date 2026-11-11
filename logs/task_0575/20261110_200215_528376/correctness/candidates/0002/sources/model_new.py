import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features

    def forward(self, x):
        x, _ = cuda_extension.max_dim1(x)
        x = cuda_extension.round_tensor(x)
        return x

    def debug_forward(self, x):
        values, indices = cuda_extension.max_dim1(x)
        yield 'step_001', (values, indices)
        rounded = cuda_extension.round_tensor(values)
        yield 'step_002', rounded
        yield 'step_003', rounded
