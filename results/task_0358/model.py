import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
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
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_max_pool2d(x, output_size=(64, 128))
        return x


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
width = 4096  # Increased from 512 to 4096 to increase computation
height = 2048  # Increased from 256 to 2048 to increase computation
stride = 1
padding = (1, 2)
dilation = (2, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]