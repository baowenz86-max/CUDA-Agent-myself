import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, output_size, in_channels, kernel_size, stride, padding):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=output_size)
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        # Store parameters as ints to avoid subscript issues in forward
        self.output_size = output_size if isinstance(output_size, int) else output_size[0]
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride if isinstance(stride, int) else stride[0]
        self.padding = padding if isinstance(padding, int) else padding[0]

    def forward(self, x):
        x = cuda_extension.adaptive_avgpool1d_forward(x, self.output_size)
        x = cuda_extension.unfold_forward(x, kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding)
        return x

    def debug_forward(self, x):
        x = cuda_extension.adaptive_avgpool1d_forward(x, self.output_size)
        yield ('step_001', x)
        x = cuda_extension.unfold_forward(x, kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding)
        yield ('step_002', x)
        yield ('step_003', x)
