import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, subtract_value):
        super().__init__()
        self.subtract_value = subtract_value

    def forward(self, x):
        return cuda_extension.subtract_scalar(x, self.subtract_value)

    def debug_forward(self, *inputs):
        x = inputs[0]
        result = cuda_extension.subtract_scalar(x, self.subtract_value)
        yield 'step_001', result
