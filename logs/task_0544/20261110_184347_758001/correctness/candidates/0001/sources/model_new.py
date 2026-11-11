import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, eps)

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.batch_norm1d_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        x = cuda_extension.batch_norm1d_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
        )
        yield 'step_002', x
        yield 'step_003', x
