import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        start_range,
        end_range,
        step,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=(0, 0),
        dilation=(1, 1),
        bias=False,
    ):
        super().__init__()
        self.arange_tensor = torch.arange(start_range, end_range, step)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x):
        x = self.conv2d(x)
        return x


batch_size = 32  # Increased from 16 to 32 to increase runtime
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
width, height = 256, 512
stride = 1
padding = (1, 2)
dilation = (2, 1)
start_range = 0
end_range = 10
step = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [start_range, end_range, step, in_channels, out_channels, kernel_size, stride, padding, dilation]