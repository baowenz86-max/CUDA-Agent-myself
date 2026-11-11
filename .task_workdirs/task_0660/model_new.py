import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_width, values):
        super().__init__()
        self.pad_width = pad_width
        self.values = values

    def forward(self, x):
        pad_left, pad_right = self.pad_width[0], self.pad_width[1]
        x = cuda_extension.constant_pad1d_forward(x, pad_left, pad_right, self.values)
        x = cuda_extension.sum_dim2_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        pad_left, pad_right = self.pad_width[0], self.pad_width[1]
        x = cuda_extension.constant_pad1d_forward(x, pad_left, pad_right, self.values)
        yield "step_001", x
        x = cuda_extension.sum_dim2_forward(x)
        yield "step_002", x
        yield "step_003", x
