import torch
import cuda_extension
from torch.nn import BatchNorm3d, Parameter


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = BatchNorm3d(10)
        self.parameter = Parameter(torch.randn(10))

    def forward(self, x):
        return cuda_extension.bn_digamma_max(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.parameter,
            self.batch_norm.eps,
        )
