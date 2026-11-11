import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.padding_size = padding_size
        # padding_size is a tuple of four ints (left, right, top, bottom)
        self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = padding_size

    def forward(self, x):
        x = cuda_extension.ceil_forward(x)
        x = cuda_extension.reflection_pad2d_forward(
            x,
            pad_left=self.pad_left,
            pad_right=self.pad_right,
            pad_top=self.pad_top,
            pad_bottom=self.pad_bottom,
        )
        return x

    def debug_forward(self, x):
        # Yield the same checkpoints as the reference, using identical custom ops.
        x = cuda_extension.ceil_forward(x)
        yield 'step_001', x
        x = cuda_extension.reflection_pad2d_forward(
            x,
            pad_left=self.pad_left,
            pad_right=self.pad_right,
            pad_top=self.pad_top,
            pad_bottom=self.pad_bottom,
        )
        yield 'step_002', x
        yield 'step_003', x
