import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = True,
    ):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.maxpool2d_forward(x, self.maxpool.kernel_size, self.maxpool.stride,
                                             self.maxpool.padding[0], self.maxpool.dilation[0])
        x = cuda_extension.relu_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.maxpool2d_forward(x, self.maxpool.kernel_size, self.maxpool.stride,
                                             self.maxpool.padding[0], self.maxpool.dilation[0])
        yield "step_001", x
        x = cuda_extension.relu_forward(x)
        yield "step_002", x
        yield "step_003", x
