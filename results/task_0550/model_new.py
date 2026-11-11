import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super().__init__()
        self.equal_op = torch.equal
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return cuda_extension.maxpool2d_forward(
            x,
            self.kernel_size,
            self.stride,
            0,
            1,
            False,
        )

    def debug_forward(self, *inputs):
        x = inputs[0]
        result = cuda_extension.maxpool2d_forward(
            x,
            self.kernel_size,
            self.stride,
            0,
            1,
            False,
        )
        yield 'step_001', result
