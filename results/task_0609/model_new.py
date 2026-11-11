import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape, mask_value):
        super().__init__()
        self.mask_value = mask_value

    def forward(self, x):
        mask = cuda_extension.isnan(x)
        x = cuda_extension.masked_fill(x, mask, self.mask_value)
        return x

    def debug_forward(self, x):
        mask = cuda_extension.isnan(x)
        yield 'step_001', mask
        x = cuda_extension.masked_fill(x, mask, self.mask_value)
        yield 'step_002', x
        yield 'step_003', x
