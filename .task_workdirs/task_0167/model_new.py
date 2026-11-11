import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, dim, max_value):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.dim = dim
        self.max_value = max_value

    def forward(self, x):
        x = cuda_extension.batch_norm_inference(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        max_vals = cuda_extension.max_reduce(x, self.dim)
        return max_vals * self.max_value

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.batch_norm_inference(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield 'step_001', x
        max_vals = cuda_extension.max_reduce(x, self.dim)
        final = max_vals * self.max_value
        yield 'step_002', final
