import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, upscale_factor, channels_factor):
        super(ModelNew, self).__init__()
        # Preserve attributes for compatibility (not used in forward)
        self.upscale_factor = upscale_factor
        self.channels_factor = channels_factor
        # Reference used nn.PixelShuffle; we keep the module for state_dict key preservation (empty parameters).
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        # Step 1: sum over channels, keepdim=True -> [N,1,H,W]
        x = cuda_extension.sum_channels_forward(x)
        # Step 2: expand to channels_factor -> [N,C',H,W]
        x = cuda_extension.expand_channels_forward(x, self.channels_factor)
        # Step 3: pixel shuffle -> [N, C'/r^2, H*r, W*r]
        x = cuda_extension.pixel_shuffle_forward(x, self.upscale_factor)
        return x

    def debug_forward(self, x):
        # Follow checkpoints: step_001, step_002, step_003, step_004 (return)
        x_sum = cuda_extension.sum_channels_forward(x)
        yield "step_001", x_sum
        x_expand = cuda_extension.expand_channels_forward(x_sum, self.channels_factor)
        yield "step_002", x_expand
        x_out = cuda_extension.pixel_shuffle_forward(x_expand, self.upscale_factor)
        yield "step_003", x_out
        yield "step_004", x_out
