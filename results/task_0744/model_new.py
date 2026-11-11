import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps, momentum):
        super().__init__()
        self.bn2d = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = cuda_extension.mean_dim0_keepdim(x)
        x = cuda_extension.batchnorm2d_eval(
            x,
            self.bn2d.weight,
            self.bn2d.bias,
            self.bn2d.running_mean,
            self.bn2d.running_var,
            self.bn2d.eps
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.mean_dim0_keepdim(x)
        yield 'step_001', x
        x = cuda_extension.batchnorm2d_eval(
            x,
            self.bn2d.weight,
            self.bn2d.bias,
            self.bn2d.running_mean,
            self.bn2d.running_var,
            self.bn2d.eps
        )
        yield 'step_002', x
        yield 'step_003', x
