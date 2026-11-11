import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        x = cuda_extension.relu6_forward(x)
        x = cuda_extension.sum_dim1_forward(x)
        x = cuda_extension.expand_forward(x, self.out_features)
        return x

    def debug_forward(self, x):
        x = cuda_extension.relu6_forward(x)
        yield 'step_001', x
        x = cuda_extension.sum_dim1_forward(x)
        yield 'step_002', x
        x = cuda_extension.expand_forward(x, self.out_features)
        yield 'step_003', x
        yield 'step_004', x
