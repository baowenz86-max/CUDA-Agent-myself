import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_features, eps):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(input_features, eps=eps)

    def forward(self, x):
        x = cuda_extension.sigmoid_forward(x)
        x = cuda_extension.batch_norm2d_inference_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.sigmoid_forward(x)
        yield 'step_001', x
        x = cuda_extension.batch_norm2d_inference_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield 'step_002', x
        yield 'step_003', x
