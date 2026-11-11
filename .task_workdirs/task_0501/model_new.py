import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import cuda_extension
        return cuda_extension.maxpool2d_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            1,  # dilation_h
            1,  # dilation_w
            False,  # ceil_mode
        )

    def debug_forward(self, *inputs):
        x = inputs[0]
        out = self.forward(x)
        yield 'step_001', out
