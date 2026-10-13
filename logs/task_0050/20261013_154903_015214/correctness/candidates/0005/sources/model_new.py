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
        bias: bool = False,
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.conv2d_forward(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.conv2d.stride,
            self.conv2d.padding,
            self.conv2d.dilation,
            self.conv2d.groups,
        )

    def debug_forward(self, x: torch.Tensor):
        out = self.forward(x)
        yield 'step_001', out
