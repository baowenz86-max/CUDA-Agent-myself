import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return cuda_extension.pixel_shuffle_forward(x, self.upscale_factor)

    def debug_forward(self, *inputs):
        x = inputs[0]
        out = cuda_extension.pixel_shuffle_forward(x, self.upscale_factor)
        yield 'step_001', out
