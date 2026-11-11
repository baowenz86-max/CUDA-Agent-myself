import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x):
        max_result = cuda_extension.max_dim0(x, self.input_shape)
        cumsum_result = cuda_extension.cumsum_dim0(x, self.input_shape)
        return max_result[0], cumsum_result

    def debug_forward(self, x):
        max_result = cuda_extension.max_dim0(x, self.input_shape)
        yield 'step_001', max_result
        cumsum_result = cuda_extension.cumsum_dim0(x, self.input_shape)
        yield 'step_002', cumsum_result
        yield 'step_003', (max_result[0], cumsum_result)
