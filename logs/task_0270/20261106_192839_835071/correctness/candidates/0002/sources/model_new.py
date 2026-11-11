import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_width, pad_mode):
        super().__init__()
        self.pad_width = pad_width
        self.pad_mode = pad_mode

    def forward(self, x):
        x = cuda_extension.custom_deg2rad(x)
        pad_l, pad_r, pad_t, pad_b = self.pad_width
        x = cuda_extension.custom_reflection_pad2d(
            x, pad_l, pad_r, pad_t, pad_b
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_deg2rad(x)
        yield 'step_001', x
        pad_l, pad_r, pad_t, pad_b = self.pad_width
        x = cuda_extension.custom_reflection_pad2d(
            x, pad_l, pad_r, pad_t, pad_b
        )
        yield 'step_002', x
        yield 'step_003', x
