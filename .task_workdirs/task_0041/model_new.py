import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.custom_neg(x)
        x = cuda_extension.custom_adaptive_avg_pool1d(x, output_size=self.pool.output_size)
        x = cuda_extension.custom_addcmul_tt(self.bias, x, x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_neg(x)
        yield 'step_001', x
        x = cuda_extension.custom_adaptive_avg_pool1d(x, output_size=self.pool.output_size)
        yield 'step_002', x
        x = cuda_extension.custom_addcmul_tt(self.bias, x, x)
        yield 'step_003', x
        yield 'step_004', x
        return x
