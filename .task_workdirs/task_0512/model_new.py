import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # The reference model stores nn.AvgPool1d in self.pool. To preserve
        # state_dict keys and initialization semantics, we emulate its presence
        # with a dummy module-like attribute that has no parameters.
        # Since state_dict is empty, no parameters are registered.
        self.pool = nn.AvgPool1d(kernel_size, stride=stride, padding=padding)
        # Store configuration for forward use
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Use custom extension operator implementing the same 1D average pooling
        y = cuda_extension.avgpool1d_forward(x, self.kernel_size, self.stride, self.padding)
        return y

    def debug_forward(self, *inputs):
        # inputs should be a single tensor x
        x = inputs[0]
        y = cuda_extension.avgpool1d_forward(x, self.kernel_size, self.stride, self.padding)
        yield 'step_001', y
        yield 'step_002', y