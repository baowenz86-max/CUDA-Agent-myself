import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.batchnorm2d = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = cuda_extension.batchnorm2d_eval_forward(
            x,
            self.batchnorm2d.weight,
            self.batchnorm2d.bias,
            self.batchnorm2d.running_mean,
            self.batchnorm2d.running_var,
            self.batchnorm2d.eps
        )
        x = cuda_extension.sum_dim_keepdim_forward(x, 1)
        return x

    def debug_forward(self, x):
        x = cuda_extension.batchnorm2d_eval_forward(
            x,
            self.batchnorm2d.weight,
            self.batchnorm2d.bias,
            self.batchnorm2d.running_mean,
            self.batchnorm2d.running_var,
            self.batchnorm2d.eps
        )
        yield 'step_001', x
        x = cuda_extension.sum_dim_keepdim_forward(x, 1)
        yield 'step_002', x
        yield 'step_003', x
