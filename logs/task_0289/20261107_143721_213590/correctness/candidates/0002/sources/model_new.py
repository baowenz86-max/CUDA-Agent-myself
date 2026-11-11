import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        # Store configuration as plain attributes; no parameters/buffers are registered.
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return cuda_extension.max_pool3d_forward(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            1,
            False,
        )

    def debug_forward(self, *inputs):
        x = inputs[0]
        out = cuda_extension.max_pool3d_forward(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            1,
            False,
        )
        yield "step_001", out
