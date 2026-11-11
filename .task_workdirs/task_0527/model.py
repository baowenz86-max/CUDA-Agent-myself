import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, kernel_size, dilation, padding):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.input_size = input_size

    def forward(self, x):
        x = self.unfold(x)
        max_val = torch.amax(x, dim=(1, 2))
        non_zero_count = torch.count_nonzero(x, dim=(1, 2))
        return max_val, non_zero_count


batch_size = 16
in_channels = 3
height, width = 512, 512  # Increased width from 32 to 512
kernel_size = (3, 3)
dilation = (1, 1)
padding = (1, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, dilation, padding]