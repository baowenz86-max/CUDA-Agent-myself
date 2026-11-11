import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = cuda_extension.upsample_bilinear2d(x, self.scale_factor, self.align_corners, False)
        x = cuda_extension.ceil_tensor(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.upsample_bilinear2d(x, self.scale_factor, self.align_corners, False)
        yield 'step_001', x
        x = cuda_extension.ceil_tensor(x)
        yield 'step_002', x
        yield 'step_003', x
