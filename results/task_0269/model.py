import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, norm_type: int, kernel_size: tuple, stride: int = 1, ceil_mode: bool = False):
        super().__init__()
        self.lplp = nn.LPPool2d(norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lplp(x)


batch_size = 16
norm_type = 2
kernel_size = (3, 5)
width = 512
height = 256
stride = 1
ceil_mode = False


def get_inputs():
    x = torch.randn(batch_size, 16, height, width)  # Increased channels from 1 to 16
    return [x]


def get_init_inputs():
    return [norm_type, kernel_size, stride, ceil_mode]