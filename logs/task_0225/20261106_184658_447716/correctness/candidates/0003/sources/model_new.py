import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x = cuda_extension.instance_norm_forward(
            x,
            self.instance_norm.weight,
            self.instance_norm.bias,
            self.instance_norm.eps
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        normalized = cuda_extension.instance_norm_forward(
            x,
            self.instance_norm.weight,
            self.instance_norm.bias,
            self.instance_norm.eps
        )
        yield 'step_001', normalized
        yield 'step_002', normalized
