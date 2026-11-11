import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, pooling_size):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride=pooling_size)

    def forward(self, x):
        y = cuda_extension.maxpool2d_forward(
            x,
            self.maxpool.kernel_size,
            self.maxpool.stride,
            self.maxpool.padding,
            self.maxpool.dilation,
            self.maxpool.ceil_mode,
        )
        x = cuda_extension.elu_forward(y, 1.0)
        return x

    def debug_forward(self, x):
        y = cuda_extension.maxpool2d_forward(
            x,
            self.maxpool.kernel_size,
            self.maxpool.stride,
            self.maxpool.padding,
            self.maxpool.dilation,
            self.maxpool.ceil_mode,
        )
        x = cuda_extension.elu_forward(y, 1.0)
        yield "step_001", x
        yield "step_002", x
