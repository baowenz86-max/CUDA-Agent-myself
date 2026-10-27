import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, num_classes, eps, momentum, affine):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        x = cuda_extension.batch_norm2d_inference(
            x, self.batch_norm.weight, self.batch_norm.bias,
            self.batch_norm.running_mean, self.batch_norm.running_var,
            self.batch_norm.eps
        )
        x = cuda_extension.leaky_relu_forward(x, 0.01)
        return x

    def debug_forward(self, x):
        x = cuda_extension.batch_norm2d_inference(
            x, self.batch_norm.weight, self.batch_norm.bias,
            self.batch_norm.running_mean, self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield 'step_001', x
        x = cuda_extension.leaky_relu_forward(x, 0.01)
        yield 'step_002', x
        yield 'step_003', x
