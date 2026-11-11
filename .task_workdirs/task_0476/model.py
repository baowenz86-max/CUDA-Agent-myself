import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad_size, roll_shifts):
        super().__init__()
        self.pad = nn.ReflectionPad2d(pad_size)
        self.roll_shifts = roll_shifts

    def forward(self, x):
        x = self.pad(x)
        x = torch.roll(x, shifts=self.roll_shifts, dims=(2, 3))
        return x


pad_size = (1, 2, 1, 2)  # padding size for each dimension
roll_shifts = (1, -1)  # number of places by which the elements of x are shifted for each dimension
batch_size = 16
in_channels = 3
height = 512  # Adjusted height to be at least 512
width = 4096  # Increased width to increase computation


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [pad_size, roll_shifts]