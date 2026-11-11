import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.convtranspose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convtranspose1d(x)
        x = torch.floor(x)
        return x


batch_size = 64
in_channels = 16
out_channels = 32
kernel_size = 5
width = 16384  # Increased from 512 to 16384 to increase computational load
stride = 2
padding = 1
output_padding = 0


def get_inputs():
    x = torch.randn(batch_size, in_channels, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]