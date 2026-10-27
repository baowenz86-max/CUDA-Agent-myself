import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_size, input_channels, input_height, input_width):
        super().__init__()
        self.pad = nn.ReflectionPad2d(pad_size)
        self.pad_size = pad_size
        self.input_height = input_height
        self.input_width = input_width

    def forward(self, x, y):
        x_padded = cuda_extension.reflection_pad2d_forward(x, self.pad_size)
        y_padded = cuda_extension.reflection_pad2d_forward(y, self.pad_size)
        max_val = cuda_extension.elementwise_max_forward(x_padded, y_padded)
        min_val = cuda_extension.elementwise_min_forward(max_val, x_padded)
        return min_val

    def debug_forward(self, x, y):
        x_padded = cuda_extension.reflection_pad2d_forward(x, self.pad_size)
        yield 'step_001', x_padded
        y_padded = cuda_extension.reflection_pad2d_forward(y, self.pad_size)
        yield 'step_002', y_padded
        max_val = cuda_extension.elementwise_max_forward(x_padded, y_padded)
        yield 'step_003', max_val
        min_val = cuda_extension.elementwise_min_forward(max_val, x_padded)
        yield 'step_004', min_val
        yield 'step_005', min_val
