import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = torch.neg(x)
        x = self.pool(x)
        x = torch.addcmul(self.bias, x, x)
        return x

    def debug_forward(self, x):
        x = torch.neg(x)
        yield 'step_001', x
        x = self.pool(x)
        yield 'step_002', x
        x = torch.addcmul(self.bias, x, x)
        yield 'step_003', x
        yield 'step_004', x
        return x
