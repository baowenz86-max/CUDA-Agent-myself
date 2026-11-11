import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        x = cuda_extension.custom_softmax(x, dim=1)
        x = cuda_extension.custom_pixel_shuffle(x, self.upscale_factor)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_softmax(x, dim=1)
        yield 'step_001', x
        x = cuda_extension.custom_pixel_shuffle(x, self.upscale_factor)
        yield 'step_002', x
        yield 'step_003', x
