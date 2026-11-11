import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, container: nn.Container):
        super().__init__()
        self.container = container

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.container(x)


batch_size = 512
in_channels = 3
out_channels = 64  # Increased from 16 to 64 to increase computation
kernel_size = 3
height, width = 128, 128  # Increased from 32x32 to 128x128 to increase input size

container = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size), torch.nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [container]