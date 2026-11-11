import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        # x is assumed to be on the correct device; if not, move it
        if not x.is_cuda:
            x = x.cuda()
        return cuda_extension.pixel_unshuffle_custom(x, self.upscale_factor)

    def debug_forward(self, x):
        # Same computation as forward but yield intermediate checkpoints
        if not x.is_cuda:
            x = x.cuda()
        out = cuda_extension.pixel_unshuffle_custom(x, self.upscale_factor)
        yield "step_001", out
