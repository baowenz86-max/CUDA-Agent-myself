import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps, momentum):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.clamp_lower_bound = 0.0
        self.clamp_upper_bound = 1.0

    def forward(self, x):
        return cuda_extension.clamp_bn2d_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )

    def debug_forward(self, x):
        step_001 = cuda_extension.clamp_forward(
            x, self.clamp_lower_bound, self.clamp_upper_bound
        )
        yield 'step_001', step_001
        step_002 = cuda_extension.bn2d_eval_forward(
            step_001,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield 'step_002', step_002
        yield 'step_003', step_002
