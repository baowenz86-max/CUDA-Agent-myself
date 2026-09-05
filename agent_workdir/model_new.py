import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm3d(10)
        self.parameter = torch.nn.Parameter(torch.randn(10))

    def forward(self, x):
        return cuda_extension.fused_bn_digamma_max(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.parameter,
        )
