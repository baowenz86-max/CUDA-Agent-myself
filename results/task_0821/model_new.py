import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, channels, pad_size, pad_mode):
        super().__init__()
        self.pad_size = pad_size
        self.pad_mode = pad_mode
        self.channels = channels

    def forward(self, x):
        # ReflectionPad2d((1,2,1,2))
        pad_left, pad_right, pad_top, pad_bottom = self.pad_size
        x = cuda_extension.reflection_pad2d_forward(x, pad_left, pad_right, pad_top, pad_bottom)
        x = cuda_extension.argsort_dim1_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        pad_left, pad_right, pad_top, pad_bottom = self.pad_size
        x = cuda_extension.reflection_pad2d_forward(x, pad_left, pad_right, pad_top, pad_bottom)
        yield "step_001", x
        x = cuda_extension.argsort_dim1_forward(x)
        yield "step_002", x
        yield "step_003", x
