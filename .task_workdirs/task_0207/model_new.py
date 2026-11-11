import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, max_channels):
        super().__init__()
        self.max_channels = max_channels

    def forward(self, x):
        x = cuda_extension.clamp_forward(x, min_val=0.0, max_val=1.0)
        x = cuda_extension.softmax_dim1_forward(x)
        x = cuda_extension.slice_channels_forward(x, self.max_channels)
        return x

    def debug_forward(self, x):
        x = cuda_extension.clamp_forward(x, min_val=0.0, max_val=1.0)
        yield 'step_001', x
        x = cuda_extension.softmax_dim1_forward(x)
        yield 'step_002', x
        x = cuda_extension.slice_channels_forward(x, self.max_channels)
        yield 'step_003', x
