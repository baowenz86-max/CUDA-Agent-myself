import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
        self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = pad

    def forward(self, x, elements):
        mask = cuda_extension.isin_forward(x, elements)
        x = cuda_extension.zero_pad2d_forward(x, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
        return x, mask

    def debug_forward(self, x, elements):
        mask = cuda_extension.isin_forward(x, elements)
        yield "step_001", mask
        x = cuda_extension.zero_pad2d_forward(x, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
        yield "step_002", x
        yield "step_003", (x, mask)
