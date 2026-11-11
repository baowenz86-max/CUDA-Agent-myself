import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps, momentum, hard_slope, scale, bias):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum)
        self.hardsigmoid = nn.Hardsigmoid(inplace=False)
        self.scale = scale
        self.bias = bias

    def forward(self, x):
        x = cuda_extension.custom_instance_norm_forward(x, self.instance_norm.eps, self.instance_norm.momentum)
        x = cuda_extension.custom_hardsigmoid_forward(x)
        x = cuda_extension.custom_scale_bias_forward(x, self.scale, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_instance_norm_forward(x, self.instance_norm.eps, self.instance_norm.momentum)
        yield 'step_001', x
        x = cuda_extension.custom_hardsigmoid_forward(x)
        yield 'step_002', x
        x = cuda_extension.custom_scale_bias_forward(x, self.scale, self.bias)
        yield 'step_003', x
        yield 'step_004', x
