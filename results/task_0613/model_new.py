import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps, momentum, affine):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        return cuda_extension.instance_norm_forward(x, self.instance_norm.weight, self.instance_norm.bias, self.instance_norm.eps)

    def debug_forward(self, *inputs):
        x = inputs[0]
        out = cuda_extension.instance_norm_forward(x, self.instance_norm.weight, self.instance_norm.bias, self.instance_norm.eps)
        yield ("step_001", out)
