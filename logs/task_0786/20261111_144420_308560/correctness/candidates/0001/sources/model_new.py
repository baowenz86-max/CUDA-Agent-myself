import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape

    def forward(self, x):
        x = cuda_extension.custom_mean(x)
        x = cuda_extension.custom_layer_norm(x, self.normalized_shape)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_mean(x)
        yield 'step_001', x
        x = cuda_extension.custom_layer_norm(x, self.normalized_shape)
        yield 'step_002', x
        yield 'step_003', x
