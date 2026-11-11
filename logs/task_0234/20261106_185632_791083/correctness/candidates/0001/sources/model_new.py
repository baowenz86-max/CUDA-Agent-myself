import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn_out = cuda_extension.batchnorm_eval_forward(
            x,
            self.batchnorm.weight,
            self.batchnorm.bias,
            self.batchnorm.running_mean,
            self.batchnorm.running_var,
            self.batchnorm.eps,
        )
        out = cuda_extension.abs_forward(bn_out)
        return out

    def debug_forward(self, *inputs):
        x = inputs[0]
        bn_out = cuda_extension.batchnorm_eval_forward(
            x,
            self.batchnorm.weight,
            self.batchnorm.bias,
            self.batchnorm.running_mean,
            self.batchnorm.running_var,
            self.batchnorm.eps,
        )
        yield ("step_001", bn_out)
        out = cuda_extension.abs_forward(bn_out)
        yield ("step_002", out)
        yield ("step_003", out)
