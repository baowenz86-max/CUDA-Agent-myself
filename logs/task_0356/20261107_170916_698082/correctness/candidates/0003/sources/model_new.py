import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, num_channels, num_groups):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.group_norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        y_bn = cuda_extension.batch_norm_inference(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
        )
        y_gn = cuda_extension.group_norm_inference(
            y_bn,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
        return y_gn

    def debug_forward(self, x):
        y_bn = cuda_extension.batch_norm_inference(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
        )
        yield "step_001", y_bn
        y_gn = cuda_extension.group_norm_inference(
            y_bn,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
        yield "step_002", y_gn
        yield "step_003", y_gn
