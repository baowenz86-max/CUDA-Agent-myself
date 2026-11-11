import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, bias_shape, hardsigmoid_slope):
        super().__init__()
        self.batch_norm2d = nn.BatchNorm2d(num_features)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardsigmoid_slope = hardsigmoid_slope

    def forward(self, x):
        x = cuda_extension.batch_norm2d_forward(
            x,
            self.batch_norm2d.weight,
            self.batch_norm2d.bias,
            self.batch_norm2d.running_mean,
            self.batch_norm2d.running_var,
            self.batch_norm2d.eps,
            False
        )
        x = cuda_extension.hardsigmoid_forward(x, self.hardsigmoid_slope)
        x = cuda_extension.add_bias_forward(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.batch_norm2d_forward(
            x,
            self.batch_norm2d.weight,
            self.batch_norm2d.bias,
            self.batch_norm2d.running_mean,
            self.batch_norm2d.running_var,
            self.batch_norm2d.eps,
            False
        )
        yield 'step_001', x
        x = cuda_extension.hardsigmoid_forward(x, self.hardsigmoid_slope)
        yield 'step_002', x
        x = cuda_extension.add_bias_forward(x, self.bias)
        yield 'step_003', x
        yield 'step_004', x
