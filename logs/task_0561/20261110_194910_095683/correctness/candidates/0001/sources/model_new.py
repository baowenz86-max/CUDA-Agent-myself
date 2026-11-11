import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Preserve state_dict key 'prelu.weight' and shape [channels]
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        # Custom op for step_001: replace NaN with zero
        x = cuda_extension.custom_nan_to_zero(x)
        # Custom op for step_002: PReLU with per-channel weight
        x = cuda_extension.custom_prelu(x, self.prelu.weight)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_nan_to_zero(x)
        yield 'step_001', x
        x = cuda_extension.custom_prelu(x, self.prelu.weight)
        yield 'step_002', x
        yield 'step_003', x
