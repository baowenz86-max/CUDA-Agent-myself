import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, num_channels, num_bins, group_num, eps, mean_shape, var_shape):
        super().__init__()
        self.group_num = group_num
        self.mean_shape = mean_shape
        self.var_shape = var_shape
        self.bins = nn.Parameter(torch.histc(torch.randn(1, num_channels, 32, 32), bins=num_bins, min=0, max=1))
        # BatchNorm2d parameters and buffers (not using the module in forward)
        self.batch_norm = nn.BatchNorm2d(num_channels, eps=eps)
        # Keep references for state_dict compatibility (automatically registered)

    def forward(self, x):
        # Step 1: batch_norm inference
        x = cuda_extension.batch_norm_inference(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        # Step 2: group_norm
        x = cuda_extension.group_norm(
            x,
            self.group_num,
            None,
            None,
            1e-05
        )
        # Step 3: histc
        x = cuda_extension.histc(
            x,
            self.bins.shape[0],
            0.0,
            1.0
        )
        return x

    def debug_forward(self, x):
        # Step 1: batch_norm inference
        x = cuda_extension.batch_norm_inference(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield 'step_001', x
        # Step 2: group_norm
        x = cuda_extension.group_norm(
            x,
            self.group_num,
            None,
            None,
            1e-05
        )
        yield 'step_002', x
        # Step 3: histc
        x = cuda_extension.histc(
            x,
            self.bins.shape[0],
            0.0,
            1.0
        )
        yield 'step_003', x
        # Final return
        yield 'step_004', x
