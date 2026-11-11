import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        channels: int,
        squeeze_dim: int,
        pad_value: int,
        pad_dims: tuple,
        kernel_size: tuple,
        stride: int = 1,
        dilation: tuple = (1, 1),
        padding: tuple = (0, 0),
        bias: bool = False,
    ):
        super().__init__()
        self.squeeze_dim = squeeze_dim
        self.squeeze = torch.squeeze
        self.pad = nn.ConstantPad2d(pad_dims, pad_value)
        self.conv2d = nn.Conv2d(
            channels, channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x, self.squeeze_dim)
        x = self.pad(x)
        return self.conv2d(x)


batch_size = 16
channels = 32  # Increased from 3 to increase computation
squeeze_dim = 0
pad_value = 0
pad_dims = (1, 1, 2, 2)  # Fixed to 4-element tuple for valid padding
kernel_size = (3, 5)
width = 512
height = 256
stride = 1
padding = (1, 2)
dilation = (2, 1)


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [channels, squeeze_dim, pad_value, pad_dims, kernel_size, stride, padding, dilation]