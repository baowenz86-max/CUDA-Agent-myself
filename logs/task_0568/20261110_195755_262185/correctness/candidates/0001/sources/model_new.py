import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_width, pad_value):
        super().__init__()
        self.pad_width = pad_width
        self.pad_value = pad_value

    def forward(self, x):
        pad_left, pad_right, pad_top, pad_bottom = self.pad_width
        x = cuda_extension.custom_pad(x, pad_left, pad_right, pad_top, pad_bottom, self.pad_value)
        x = cuda_extension.custom_max_dim1(x)
        return x

    def debug_forward(self, x):
        pad_left, pad_right, pad_top, pad_bottom = self.pad_width
        pad_out = cuda_extension.custom_pad(x, pad_left, pad_right, pad_top, pad_bottom, self.pad_value)
        max_out = cuda_extension.custom_max_dim1(pad_out)
        yield 'step_001', pad_out
        yield 'step_002', max_out
        yield 'step_003', max_out
