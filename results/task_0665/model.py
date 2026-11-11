import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0)):
        super().__init__()
        self.mean = torch.mean
        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mean(x, dim=1, keepdim=True)  # mean across channels
        x = self.max_pool(x)
        return x


batch_size = 16
channels = 3
kernel_size = (2, 2)
width = 4096  # Increased from 512 to 4096 to increase runtime
height = 2048  # Increased from 256 to 2048 to increase runtime
stride = 2
padding = (0, 0)


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [channels, kernel_size, stride, padding]