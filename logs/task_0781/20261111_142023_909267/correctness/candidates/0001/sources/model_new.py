import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, channels, padding_size):
        super().__init__()
        self.channels = channels
        self.padding_size = padding_size

    def forward(self, x):
        return cuda_extension.custom_pad_mean(x, self.padding_size, self.channels)

    def debug_forward(self, x):
        padded = cuda_extension.custom_pad(x, self.padding_size)
        yield 'step_001', padded
        mean = cuda_extension.custom_mean(padded, self.channels)
        yield 'step_002', mean
        yield 'step_003', mean
        return mean
