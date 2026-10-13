import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.instance_norm3d_forward(x, self.num_features, self.eps)
        return x

    def debug_forward(self, x):
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        x = cuda_extension.instance_norm3d_forward(x, self.num_features, self.eps)
        yield 'step_002', x
        yield 'step_003', x
