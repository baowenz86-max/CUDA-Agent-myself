import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, values_shape):
        super().__init__()
        # MaxPool2d parameters stored for later use
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Parameter to match state_dict key 'values'
        self.values = nn.Parameter(torch.randn(values_shape))

    def forward(self, x):
        # step 1: mask = torch.isin(x, self.values) -> custom op
        mask = cuda_extension.isin_forward(x, self.values)
        # step 2: x = self.maxpool(x) -> custom maxpool using cuDNN
        x = cuda_extension.maxpool2d_forward(
            x, self.kernel_size, self.stride, self.padding
        )
        return x

    def debug_forward(self, x):
        # Same computation, yielding checkpoints
        mask = cuda_extension.isin_forward(x, self.values)
        yield ('step_001', mask)
        x = cuda_extension.maxpool2d_forward(
            x, self.kernel_size, self.stride, self.padding
        )
        yield ('step_002', x)
        yield ('step_003', x)
