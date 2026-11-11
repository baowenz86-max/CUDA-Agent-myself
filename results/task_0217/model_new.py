import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x):
        x = cuda_extension.custom_log_softmax(x)
        x = cuda_extension.custom_mul_scalar(x, self.multiplier)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_log_softmax(x)
        yield 'step_001', x
        x = cuda_extension.custom_mul_scalar(x, self.multiplier)
        yield 'step_002', x
        yield 'step_003', x
        return x
