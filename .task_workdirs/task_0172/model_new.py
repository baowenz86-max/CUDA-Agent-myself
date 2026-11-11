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
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.triu_last2d(x)
        return cuda_extension.conv2d(x, self.conv2d.weight)

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.triu_last2d(x)
        yield 'step_001', x
        x = cuda_extension.conv2d(x, self.conv2d.weight)
        yield 'step_002', x
