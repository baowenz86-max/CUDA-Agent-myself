import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, output_size, in_channels, kernel_size, stride, padding):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=output_size)
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = cuda_extension.adaptive_avgpool1d_forward(x, self.avgpool.output_size[0])
        x = cuda_extension.unfold_forward(x, kernel_size=self.unfold.kernel_size[0],
                                          stride=self.unfold.stride[0],
                                          padding=self.unfold.padding[0])
        return x

    def debug_forward(self, x):
        x = cuda_extension.adaptive_avgpool1d_forward(x, self.avgpool.output_size[0])
        yield ('step_001', x)
        x = cuda_extension.unfold_forward(x, kernel_size=self.unfold.kernel_size[0],
                                          stride=self.unfold.stride[0],
                                          padding=self.unfold.padding[0])
        yield ('step_002', x)
        yield ('step_003', x)
