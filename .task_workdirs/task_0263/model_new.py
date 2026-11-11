import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        self.in_channels = in_channels

    def forward(self, x):
        x_avg, x_max = cuda_extension.pool2d_avgmax(
            x, self.kernel_size[0], self.kernel_size[1]
        )
        return x_avg, x_max

    def debug_forward(self, x):
        x_avg, x_max = cuda_extension.pool2d_avgmax(
            x, self.kernel_size[0], self.kernel_size[1]
        )
        yield "step_001", x_avg
        yield "step_002", x_max
        yield "step_003", (x_avg, x_max)
