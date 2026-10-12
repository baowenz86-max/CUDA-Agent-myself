import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, kernel_size, dilation, padding, unfolded_channels):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.unfolded_channels = unfolded_channels

    def forward(self, x):
        x = cuda_extension.unfold_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.dilation[0],
            self.dilation[1],
            self.padding[0],
            self.padding[1],
            1,
            1,
        )
        x = cuda_extension.sign_forward(x)
        x = cuda_extension.cos_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.unfold_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.dilation[0],
            self.dilation[1],
            self.padding[0],
            self.padding[1],
            1,
            1,
        )
        yield "step_001", x
        x = cuda_extension.sign_forward(x)
        yield "step_002", x
        x = cuda_extension.cos_forward(x)
        yield "step_003", x
        yield "step_004", x
