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
        dilation: tuple = (1, 1),
        bias: bool = True,
    ):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = torch.relu(x)
        return x


batch_size = 64
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
width = 512
height = 512  # Increased from 256 to 512 to match width dimension
stride = 1
padding = (1, 2)
dilation = (2, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]