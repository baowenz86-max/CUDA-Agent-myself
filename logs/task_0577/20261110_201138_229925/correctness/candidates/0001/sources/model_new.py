import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape, eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_shape, eps=eps)

    def forward(self, x):
        x = cuda_extension.custom_abs(x)
        x = cuda_extension.custom_layer_norm(x, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_abs(x)
        yield 'step_001', x
        x = cuda_extension.custom_layer_norm(x, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps)
        yield 'step_002', x
        yield 'step_003', x
