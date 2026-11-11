import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, has_nan):
        super().__init__()
        self.has_nan = has_nan

    def forward(self, x):
        if self.has_nan:
            x = cuda_extension.custom_isnan(x)
        else:
            x = cuda_extension.custom_relu6(x)
        return x

    def debug_forward(self, x):
        if self.has_nan:
            y = cuda_extension.custom_isnan(x)
            yield 'output', y
        else:
            y = cuda_extension.custom_relu6(x)
            yield 'output', y
