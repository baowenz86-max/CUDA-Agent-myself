import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        bn_out = cuda_extension.batchnorm2d_eval_forward(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps
        )
        x = cuda_extension.relu_forward(bn_out)
        return x

    def debug_forward(self, x):
        bn_out = cuda_extension.batchnorm2d_eval_forward(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps
        )
        yield 'step_001', bn_out
        x = cuda_extension.relu_forward(bn_out)
        yield 'step_002', x
        yield 'step_003', x
