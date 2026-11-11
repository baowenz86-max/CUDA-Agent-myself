import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, pooling_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.pooling_ratio = pooling_ratio[0]
        # No trainable parameters; maxpool3d is stateless in evaluation mode.
        # We store kernel_size for the custom pooling operator.
        self.kernel_size = kernel_size[0]

    def forward(self, x):
        # Step 1: sum over channel dimension (dim=1), keepdim=True
        x = cuda_extension.sum_dim1_keepdim(x)
        # Step 2: 3D max pooling with stored kernel_size and pooling_ratio,
        # padding=0, dilation=1, ceil_mode=False (defaults)
        x = cuda_extension.maxpool3d(x, self.kernel_size, self.pooling_ratio)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.sum_dim1_keepdim(x)
        yield 'step_001', x
        x = cuda_extension.maxpool3d(x, self.kernel_size, self.pooling_ratio)
        yield 'step_002', x
        yield 'step_003', x
