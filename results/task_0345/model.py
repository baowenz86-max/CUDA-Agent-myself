import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_channels: int, kernel_size: tuple, dilation: tuple, bias: bool, num_features: int):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation)
        x = self.bn(x)
        return x


batch_size = 16
num_channels = 3
height, width = 512, 512  # Increased width to 512
kernel_size = (3, 3)
dilation = (2, 2)
bias = False
num_features = num_channels * (kernel_size[0] * kernel_size[1])


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_channels, kernel_size, dilation, bias, num_features]