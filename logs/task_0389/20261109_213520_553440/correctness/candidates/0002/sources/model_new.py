import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        x = cuda_extension.batch_norm2d_eval(
            x,
            self.batchnorm.weight,
            self.batchnorm.bias,
            self.batchnorm.running_mean,
            self.batchnorm.running_var,
            self.batchnorm.eps,
        )
        x = cuda_extension.softmax_channel(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.batch_norm2d_eval(
            x,
            self.batchnorm.weight,
            self.batchnorm.bias,
            self.batchnorm.running_mean,
            self.batchnorm.running_var,
            self.batchnorm.eps,
        )
        yield "step_001", x
        x = cuda_extension.softmax_channel(x)
        yield "step_002", x
        yield "step_003", x
