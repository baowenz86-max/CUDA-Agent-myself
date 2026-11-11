import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_tensor_shape, elements_shape, split_sizes):
        super().__init__()
        self.elements = nn.Parameter(torch.randn(elements_shape))
        self.split_sizes = split_sizes

    def forward(self, x):
        mask = cuda_extension.isin_custom(x, self.elements)
        parts = cuda_extension.split_with_sizes_custom(x, self.split_sizes)
        return mask, parts

    def debug_forward(self, x):
        mask = cuda_extension.isin_custom(x, self.elements)
        yield 'step_001', mask
        parts = cuda_extension.split_with_sizes_custom(x, self.split_sizes)
        yield 'step_002', parts
        yield 'step_003', (mask, parts)
