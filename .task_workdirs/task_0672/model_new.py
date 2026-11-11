import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, p):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        x = cuda_extension.custom_cat_forward(x1, x2, dim=1)
        x = cuda_extension.custom_normalize_forward(x, p=self.p, dim=1)
        return x

    def debug_forward(self, x1, x2):
        x = cuda_extension.custom_cat_forward(x1, x2, dim=1)
        yield 'step_001', x
        x = cuda_extension.custom_normalize_forward(x, p=self.p, dim=1)
        yield 'step_002', x
        yield 'step_003', x
