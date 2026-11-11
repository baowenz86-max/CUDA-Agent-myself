import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(upscale_factor)

    def forward(self, x):
        return cuda_extension.pixel_unshuffle_forward(x, self.pixel_unshuffle.downscale_factor)

    def debug_forward(self, x):
        output = cuda_extension.pixel_unshuffle_forward(x, self.pixel_unshuffle.downscale_factor)
        yield 'step_001', output
