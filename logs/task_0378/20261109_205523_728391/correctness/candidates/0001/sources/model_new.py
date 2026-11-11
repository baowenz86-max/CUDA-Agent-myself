import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x):
        weight = self.batch_norm.weight if self.batch_norm.affine else None
        bias = self.batch_norm.bias if self.batch_norm.affine else None
        return cuda_extension.batch_norm_eval_forward(
            x, weight, bias, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.eps
        )

    def debug_forward(self, x):
        result = self.forward(x)
        yield ("step_001", result)
        yield ("final_return", result)
