import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.batch_norm_2d = nn.BatchNorm2d(in_features)

    def forward(self, x):
        x = cuda_extension.batch_norm_2d_eval(
            x,
            self.batch_norm_2d.weight,
            self.batch_norm_2d.bias,
            self.batch_norm_2d.running_mean,
            self.batch_norm_2d.running_var,
            1e-05
        )
        x = cuda_extension.clamp_forward(x, 0.0, 1.0)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.batch_norm_2d_eval(
            x,
            self.batch_norm_2d.weight,
            self.batch_norm_2d.bias,
            self.batch_norm_2d.running_mean,
            self.batch_norm_2d.running_var,
            1e-05
        )
        yield ('step_001', x)
        x = cuda_extension.clamp_forward(x, 0.0, 1.0)
        yield ('step_002', x)
        yield ('step_003', x)
