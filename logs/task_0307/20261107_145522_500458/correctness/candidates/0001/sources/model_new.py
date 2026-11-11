import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, upscale_factor, mode="bilinear", align_corners=None):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = cuda_extension.gelu_forward(x)
        x = cuda_extension.pixel_shuffle_forward(x, self.upscale_factor)
        x = cuda_extension.upsample_bilinear2x_forward(x, self.align_corners)
        return x

    def debug_forward(self, x):
        x = cuda_extension.gelu_forward(x)
        yield "step_001", x
        x = cuda_extension.pixel_shuffle_forward(x, self.upscale_factor)
        yield "step_002", x
        x = cuda_extension.upsample_bilinear2x_forward(x, self.align_corners)
        yield "step_003", x
        yield "step_004", x
