import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, num_channels, num_groups):
        super().__init__()
        # Use actual BatchNorm2d and GroupNorm to get exact state_dict keys and behavior.
        # They are not called in forward; instead we use custom ops with their parameters/buffers.
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.group_norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        x = cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.eps,
            self.group_norm.num_groups,
            self.batch_norm.num_features  # This is C, but pass input C; better to use self.group_norm.num_channels if available
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield 'step_001', x
        x = cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.eps,
            self.group_norm.num_groups,
            self.batch_norm.num_features
        )
        yield 'step_002', x
        yield 'step_003', x
