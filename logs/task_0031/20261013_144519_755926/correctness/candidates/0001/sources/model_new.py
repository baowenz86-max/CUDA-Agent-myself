import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, num_bins, group_num, eps, mean_shape, var_shape):
        super().__init__()
        # Match original Model structure and state_dict keys
        self.batch_norm = nn.BatchNorm2d(num_channels, eps=eps)
        self.bins = nn.Parameter(torch.histc(torch.randn(1, num_channels, 32, 32), bins=num_bins, min=0, max=1))
        self.group_num = group_num
        self.mean_shape = mean_shape
        self.var_shape = var_shape
        self.eps = eps

    def forward(self, x):
        # Step 1: BatchNorm2d inference
        x = cuda_extension.batch_norm_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        # Step 2: GroupNorm without affine
        x = cuda_extension.group_norm_forward(
            x,
            self.group_num,
            self.eps
        )
        # Step 3: Histogram, bins from self.bins.shape[0]
        x = cuda_extension.histc_forward(
            x,
            self.bins.shape[0],
            0.0,
            1.0
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]

        # Step 1
        x = cuda_extension.batch_norm_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield "step_001", x

        # Step 2
        x = cuda_extension.group_norm_forward(
            x,
            self.group_num,
            self.eps
        )
        yield "step_002", x

        # Step 3
        x = cuda_extension.histc_forward(
            x,
            self.bins.shape[0],
            0.0,
            1.0
        )
        yield "step_003", x

        # Final return
        yield "step_004", x
        return
