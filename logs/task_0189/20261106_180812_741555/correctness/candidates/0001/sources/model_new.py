import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        return cuda_extension.adaptive_avg_pool3d_2(x, self.out_channels)

    def debug_forward(self, x):
        y = self.forward(x)
        yield 'step_001', y
