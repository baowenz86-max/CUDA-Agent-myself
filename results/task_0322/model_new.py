import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return cuda_extension.channel_shuffle_cuda(x, self.groups)

    def debug_forward(self, x):
        out = cuda_extension.channel_shuffle_cuda(x, self.groups)
        yield 'step_001', out
        yield 'step_002', out
