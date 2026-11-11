import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        # Ensure input is on CUDA and contiguous
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        return cuda_extension.upsample_bilinear_2d(x, self.scale_factor)

    def debug_forward(self, x):
        y = self.forward(x)
        yield 'step_001', y
