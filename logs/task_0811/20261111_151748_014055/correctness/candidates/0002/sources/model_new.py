import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode, softmax_dim):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.softmax_dim = softmax_dim

    def forward(self, x):
        x = cuda_extension.custom_upsample_bilinear(x, self.scale_factor)
        x = cuda_extension.custom_softmax_dim(x, self.softmax_dim)
        return x

    def debug_forward(self, x):
        upsample_out = cuda_extension.custom_upsample_bilinear(x, self.scale_factor)
        yield 'step_001', upsample_out
        softmax_out = cuda_extension.custom_softmax_dim(upsample_out, self.softmax_dim)
        yield 'step_002', softmax_out
        yield 'step_003', softmax_out
