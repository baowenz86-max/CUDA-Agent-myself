import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_channels, kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.input_channels = input_channels
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = cuda_extension.log_softmax_forward(x)
        x = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        return x

    def debug_forward(self, x):
        x = cuda_extension.log_softmax_forward(x)
        yield 'step_001', x
        x = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        yield 'step_002', x
        yield 'step_003', x
