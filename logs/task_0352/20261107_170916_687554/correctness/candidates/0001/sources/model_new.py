import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = cuda_extension.upsample_bilinear2d(
            x, scale_factor=self.scale_factor
        )
        x = cuda_extension.reflection_pad2d(
            x, padding_left=1, padding_right=1, padding_top=1, padding_bottom=1
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.upsample_bilinear2d(
            x, scale_factor=self.scale_factor
        )
        yield 'step_001', x
        x = cuda_extension.reflection_pad2d(
            x, padding_left=1, padding_right=1, padding_top=1, padding_bottom=1
        )
        yield 'step_002', x
        yield 'step_003', x
