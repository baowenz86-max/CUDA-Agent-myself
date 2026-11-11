import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
    ):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.hardtanh(x)
        x = torch.mean(x, dim=1, keepdim=True)
        return x


batch_size = 64
in_channels = 12
out_channels = 64
kernel_size = (3, 5)
width = 1024  # Increased from 512 to 1024 to increase computation
height = 1024  # Increased from 256 to 1024 to increase computation
stride = 1
padding = (1, 2)
dilation = (2, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]