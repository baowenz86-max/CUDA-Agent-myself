import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = False,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = torch.unsqueeze(x, 1)  # unsqueeze at dim=1
        return x


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
stride = 2
padding = (1, 2)
output_padding = (1, 0)
dilation = (2, 1)
height = 512  # Added hyperparameter for input height
width = 64  # Increased width from 5 to 64 to increase computation


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation]