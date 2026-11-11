import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = cuda_extension.upsample_bilinear2d_forward(x, self.scale_factor)
        x = cuda_extension.reflection_pad2d_forward(x, torch.tensor([1, 1, 1, 1], dtype=torch.int64))
        return x

    def debug_forward(self, x):
        x = cuda_extension.upsample_bilinear2d_forward(x, self.scale_factor)
        yield "step_001", x
        x = cuda_extension.reflection_pad2d_forward(x, torch.tensor([1, 1, 1, 1], dtype=torch.int64))
        yield "step_002", x
        yield "step_003", x
