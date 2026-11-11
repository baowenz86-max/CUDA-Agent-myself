import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, upscale_factor, out_channels):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.out_channels = out_channels
        # No learnable parameters, state_dict is empty like reference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.abs_forward(x)
        x = cuda_extension.pixel_shuffle_forward(x, self.upscale_factor)
        return x

    def debug_forward(self, x: torch.Tensor):
        x_abs = cuda_extension.abs_forward(x)
        yield 'step_001', x_abs
        x_ps = cuda_extension.pixel_shuffle_forward(x_abs, self.upscale_factor)
        yield 'step_002', x_ps
        yield 'step_003', x_ps
