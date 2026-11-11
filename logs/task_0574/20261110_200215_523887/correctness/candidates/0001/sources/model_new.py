import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        # Move input to CUDA if necessary. This is not a computational operation.
        x = x.cuda()
        # Check for NaNs in the input tensor and return the indices
        nan_indices = cuda_extension.isnan_forward(x)
        # Apply batch normalization using custom CUDA operator
        x = cuda_extension.batch_norm_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.training,
            self.batch_norm.eps,
            self.batch_norm.momentum
        )
        return x, nan_indices

    def debug_forward(self, x):
        x = x.cuda()
        # checkpoint step_001
        nan_indices = cuda_extension.isnan_forward(x)
        yield 'step_001', nan_indices
        # checkpoint step_002
        x = cuda_extension.batch_norm_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.training,
            self.batch_norm.eps,
            self.batch_norm.momentum
        )
        yield 'step_002', x
        # checkpoint step_003 (final return)
        yield 'step_003', (x, nan_indices)
