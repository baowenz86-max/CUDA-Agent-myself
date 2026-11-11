import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        x = cuda_extension.custom_abs_forward(x)
        x = cuda_extension.custom_pad_forward(
            x,
            pad_left=1,
            pad_right=2,
            pad_top=3,
            pad_bottom=4,
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_abs_forward(x)
        yield 'step_001', x
        x = cuda_extension.custom_pad_forward(
            x,
            pad_left=1,
            pad_right=2,
            pad_top=3,
            pad_bottom=4,
        )
        yield 'step_002', x
        yield 'step_003', x
