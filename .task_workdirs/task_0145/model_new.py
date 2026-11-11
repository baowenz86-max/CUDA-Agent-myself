import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x = cuda_extension.tril_forward(x, diagonal=0)
        return cuda_extension.instance_norm_eval_forward(
            x,
            self.instance_norm.running_mean,
            self.instance_norm.running_var,
            self.instance_norm.eps
        )

    def debug_forward(self, *inputs):
        x = inputs[0]
        x_tril = cuda_extension.tril_forward(x, diagonal=0)
        yield 'step_001', x_tril
        y = cuda_extension.instance_norm_eval_forward(
            x_tril,
            self.instance_norm.running_mean,
            self.instance_norm.running_var,
            self.instance_norm.eps
        )
        yield 'step_002', y
