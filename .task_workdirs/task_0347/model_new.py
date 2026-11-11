import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, batch_size, in_channels, out_channels, kernel_size, dilation, height, width):
        super().__init__()
        # Store configuration for forward, but do not use torch.nn.Unfold
        self.kernel_size = kernel_size
        self.dilation = dilation
        # No parameters, state_dict stays empty

    def forward(self, x):
        # Call custom CUDA extension unfold, reproducing nn.Unfold with stride=1, padding=0
        # kernel_size and dilation are tuples, pass as ints
        x = cuda_extension.unfold(x, self.kernel_size[0], self.kernel_size[1],
                                  self.dilation[0], self.dilation[1])
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        # Step 001: unfold
        x = cuda_extension.unfold(x, self.kernel_size[0], self.kernel_size[1],
                                  self.dilation[0], self.dilation[1])
        yield 'step_001', x
        # Step 002: return x
        yield 'step_002', x
