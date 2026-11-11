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
        bn = self.batch_norm
        return cuda_extension.batch_norm_inference(
            x, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
        )

    def debug_forward(self, x):
        out = self.forward(x)
        yield 'step_001', out
