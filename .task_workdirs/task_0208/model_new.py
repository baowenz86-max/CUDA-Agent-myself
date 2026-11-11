import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_features)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = cuda_extension.custom_batchnorm2d_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        x = cuda_extension.custom_log_softmax_dim1_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_batchnorm2d_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield 'step_001', x
        x = cuda_extension.custom_log_softmax_dim1_forward(x)
        yield 'step_002', x
        yield 'step_003', x
