import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, num_features, eps):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        x = cuda_extension.custom_hardsigmoid(x)
        x = cuda_extension.custom_instance_norm2d_no_affine(x, self.eps)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_hardsigmoid(x)
        yield 'step_001', x
        x = cuda_extension.custom_instance_norm2d_no_affine(x, self.eps)
        yield 'step_002', x
        yield 'step_003', x
