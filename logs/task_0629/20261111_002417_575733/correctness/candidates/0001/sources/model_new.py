import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_inputs, num_outputs, multiplier_shape):
        super().__init__()
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, *inputs):
        concatenated = cuda_extension.custom_cat_forward(list(inputs), dim=1)
        multiplied = cuda_extension.custom_mul_forward(concatenated, self.multiplier)
        return multiplied

    def debug_forward(self, *inputs):
        concatenated = cuda_extension.custom_cat_forward(list(inputs), dim=1)
        yield 'step_001', concatenated
        multiplied = cuda_extension.custom_mul_forward(concatenated, self.multiplier)
        yield 'step_002', multiplied
        yield 'step_003', multiplied
