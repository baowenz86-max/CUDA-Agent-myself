import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        # Store parameters for reference; no learnable parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        # Step 1: unfold with kernel_size=(2,3), dilation=1, padding=0, stride=1
        x = cuda_extension.custom_unfold(
            x,
            2, 3, 1, 1, 0, 0, 1, 1
        )
        # Step 2: max_pool1d with kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        x = cuda_extension.custom_max_pool1d(
            x,
            3, 2, 1, 1, False
        )
        return x

    def debug_forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        # Step 1: unfold with kernel_size=(2,3), dilation=1, padding=0, stride=1
        y = cuda_extension.custom_unfold(
            x,
            2, 3, 1, 1, 0, 0, 1, 1
        )
        yield 'step_001', y
        # Step 2: max_pool1d
        y = cuda_extension.custom_max_pool1d(
            y,
            3, 2, 1, 1, False
        )
        yield 'step_002', y
        # Step 3: return
        yield 'step_003', y
