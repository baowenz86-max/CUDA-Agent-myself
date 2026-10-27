import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.max_pool1d = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = F.unfold(x, kernel_size=(2, 3))
        x = self.max_pool1d(x)
        return x


batch_size = 512
in_channels = 16
time_steps = 2048  # Increased from 128 to increase computation
kernel_size = 3
stride = 2
padding = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, time_steps)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]