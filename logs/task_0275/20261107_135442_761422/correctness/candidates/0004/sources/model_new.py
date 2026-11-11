import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(num_features)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = cuda_extension.custom_batchnorm2d(
            x,
            self.batchnorm.weight,
            self.batchnorm.bias,
            self.batchnorm.running_mean,
            self.batchnorm.running_var,
            self.batchnorm.eps,
            self.batchnorm.training,
        )
        x = cuda_extension.custom_gelu(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_batchnorm2d(
            x,
            self.batchnorm.weight,
            self.batchnorm.bias,
            self.batchnorm.running_mean,
            self.batchnorm.running_var,
            self.batchnorm.eps,
            self.batchnorm.training,
        )
        yield "step_001", x
        x = cuda_extension.custom_gelu(x)
        yield "step_002", x
        yield "step_003", x
