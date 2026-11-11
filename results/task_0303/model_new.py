import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return cuda_extension.pixel_shuffle(x, self.scale_factor)

    def debug_forward(self, *inputs):
        x = inputs[0]
        output = cuda_extension.pixel_shuffle(x, self.scale_factor)
        yield 'step_001', output
