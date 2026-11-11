import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The reference constructor stores no parameters or buffers.

    def forward(self, x):
        # Step 1: sum along channel dimension (dim=1)
        x = cuda_extension.channel_sum(x)
        # Step 2: element-wise exponential
        x = cuda_extension.exp(x)
        return x

    def debug_forward(self, x):
        # Debug path mirrors the reference computation exactly.
        sum_result = cuda_extension.channel_sum(x)
        yield 'step_001', sum_result

        exp_result = cuda_extension.exp(sum_result)
        yield 'step_002', exp_result

        yield 'step_003', exp_result
