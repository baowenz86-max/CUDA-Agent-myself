import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features

    def forward(self, x):
        x = cuda_extension.custom_mean_dim1(x)
        x = cuda_extension.custom_leaky_relu(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_mean_dim1(x)
        yield 'step_001', x
        x = cuda_extension.custom_leaky_relu(x)
        yield 'step_002', x
        yield 'step_003', x
