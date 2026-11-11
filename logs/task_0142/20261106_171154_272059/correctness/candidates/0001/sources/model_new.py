import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, batchnorm_num_features, pooling_kernel_size):
        super().__init__()
        self.batchnorm2d = nn.BatchNorm2d(batchnorm_num_features)
        self.pooling = nn.AvgPool2d(pooling_kernel_size)

    def forward(self, x):
        x = cuda_extension.batchnorm2d_forward(
            x,
            self.batchnorm2d.weight,
            self.batchnorm2d.bias,
            self.batchnorm2d.running_mean,
            self.batchnorm2d.running_var,
            self.batchnorm2d.eps,
        )
        x = cuda_extension.square_forward(x)
        x = cuda_extension.avgpool2d_forward(
            x,
            self.pooling.kernel_size,
            self.pooling.stride,
            self.pooling.padding,
            False,
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.batchnorm2d_forward(
            x,
            self.batchnorm2d.weight,
            self.batchnorm2d.bias,
            self.batchnorm2d.running_mean,
            self.batchnorm2d.running_var,
            self.batchnorm2d.eps,
        )
        yield 'step_001', x
        x = cuda_extension.square_forward(x)
        yield 'step_002', x
        x = cuda_extension.avgpool2d_forward(
            x,
            self.pooling.kernel_size,
            self.pooling.stride,
            self.pooling.padding,
            False,
        )
        yield 'step_003', x
        yield 'step_004', x
