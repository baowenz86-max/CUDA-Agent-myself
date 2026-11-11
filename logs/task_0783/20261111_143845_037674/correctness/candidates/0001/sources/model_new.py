import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, mask_value):
        super().__init__()
        self.mask_value = mask_value

    def forward(self, x):
        x = cuda_extension.custom_masked_fill(x, self.mask_value)
        x = cuda_extension.custom_cat(x, x, 1)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_masked_fill(x, self.mask_value)
        yield 'step_001', x
        x = cuda_extension.custom_cat(x, x, 1)
        yield 'step_002', x
        yield 'step_003', x
