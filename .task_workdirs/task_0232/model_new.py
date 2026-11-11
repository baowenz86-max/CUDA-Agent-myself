import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x):
        x = cuda_extension.instance_norm_forward(x, self.instance_norm.weight, self.instance_norm.bias, self.instance_norm.eps)
        x = cuda_extension.mul_scalar(x, 2.0)
        return x

    def debug_forward(self, x):
        x_after_norm = cuda_extension.instance_norm_forward(x, self.instance_norm.weight, self.instance_norm.bias, self.instance_norm.eps)
        yield "step_001", x_after_norm
        x_after_mul = cuda_extension.mul_scalar(x_after_norm, 2.0)
        yield "step_002", x_after_mul
        yield "step_003", x_after_mul
