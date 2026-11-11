import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = cuda_extension.adaptive_avg_pool1d(x, self.out_features)
        return x

    def debug_forward(self, x):
        x = cuda_extension.adaptive_avg_pool1d(x, self.out_features)
        yield 'step_001', x
        yield 'step_002', x
