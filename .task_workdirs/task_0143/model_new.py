import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        x = cuda_extension.clamp_hardswish_forward(x)
        return x

    def debug_forward(self, x):
        # checkpoint 1: after clamp
        clamped = cuda_extension.clamp_forward(x)
        yield 'step_001', clamped
        # checkpoint 2: after hardswish
        final = cuda_extension.clamp_hardswish_forward(x)
        yield 'step_002', final
        # checkpoint 3: final return
        yield 'step_003', final
