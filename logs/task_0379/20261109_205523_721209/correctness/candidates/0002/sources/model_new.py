import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, x):
        x = cuda_extension.custom_max_dim1(x)
        x = cuda_extension.custom_exp(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_max_dim1(x)
        yield 'step_001', x
        x = cuda_extension.custom_exp(x)
        yield 'step_002', x
        yield 'step_003', x
