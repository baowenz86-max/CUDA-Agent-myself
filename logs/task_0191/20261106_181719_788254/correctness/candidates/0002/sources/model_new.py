import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps, affine=True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps, affine=affine)

    def forward(self, x):
        x = cuda_extension.instance_norm_forward(
            x, self.instance_norm.weight, self.instance_norm.bias, self.instance_norm.eps
        )
        x = cuda_extension.abs_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.instance_norm_forward(
            x, self.instance_norm.weight, self.instance_norm.bias, self.instance_norm.eps
        )
        yield 'step_001', x
        x = cuda_extension.abs_forward(x)
        yield 'step_002', x
        yield 'step_003', x
