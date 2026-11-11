import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_result = cuda_extension.custom_max_dim1(x)
        max_values = max_result[0]
        rolled = cuda_extension.custom_roll_dim2(max_values)
        return rolled

    def debug_forward(self, x):
        max_result = cuda_extension.custom_max_dim1(x)
        yield 'step_001', max_result
        max_values = max_result[0]
        rolled = cuda_extension.custom_roll_dim2(max_values)
        yield 'step_002', rolled
        yield 'step_003', rolled
