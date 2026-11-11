import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.padding_size = padding_size

    def forward(self, x):
        pad_left, pad_right, pad_top, pad_bottom = self.padding_size
        return cuda_extension.zero_pad_forward(x, pad_top, pad_bottom, pad_left, pad_right)

    def debug_forward(self, *inputs):
        x = inputs[0]
        pad_left, pad_right, pad_top, pad_bottom = self.padding_size
        out = cuda_extension.zero_pad_forward(x, pad_top, pad_bottom, pad_left, pad_right)
        yield ('step_001', out)
