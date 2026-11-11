import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps, momentum, affine, track_running_stats):
        super().__init__()
        self.instance_norm = nn.InstanceNorm3d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        return cuda_extension.instance_norm3d_forward(
            x,
            self.instance_norm.running_mean,
            self.instance_norm.running_var,
            self.instance_norm.eps,
        )

    def debug_forward(self, *inputs):
        x = inputs[0]
        y = self.forward(x)
        yield 'step_001', y
