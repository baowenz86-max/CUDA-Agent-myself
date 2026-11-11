import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.maxpool1d = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool1d(x)


batch_size = 256
in_channels = 128
kernel_size = 3
width = 8192  # Increased from 512 to 8192 to increase computation


def get_inputs():
    x = torch.randn(batch_size, in_channels, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size]