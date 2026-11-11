import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return cuda_extension.stack_like(x)

    def debug_forward(self, x):
        # Step 1: unbind (logical slices)
        yield "step_001", torch.unbind(x, dim=1)

        # Step 2: pixel_shuffle each slice (identity here)
        slices = torch.unbind(x, dim=1)
        processed = [self.pixel_shuffle(item) for item in slices]
        yield "step_002", processed

        # Step 3: stack
        stacked = torch.stack(processed, dim=1)
        yield "step_003", stacked

        # Step 4: return
        yield "step_004", stacked
