import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x):
        x_norm = cuda_extension.batch_norm_1d_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        x_equal = torch.equal(x, x_norm)
        return x_equal, x_norm

    def debug_forward(self, *inputs):
        x = inputs[0]
        x_norm = cuda_extension.batch_norm_1d_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield ('step_001', x_norm)
        x_equal = torch.equal(x, x_norm)
        yield ('step_002', x_equal)
        yield ('step_003', (x_equal, x_norm))
