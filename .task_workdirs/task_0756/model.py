import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int,
        padding: tuple,
        dilation: tuple,
        groups: int,
        bias: bool,
    ):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        return x


# Hyperparameters determining test input shape
batch_size = 512
in_channels = 10
height = 256  # Increased from 64
width = 256   # Increased from 64


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    in_channels = 10
    out_channels = 64
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1
    bias = False
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias]