import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, padding_size):
        super().__init__()
        self.softplus = nn.Softplus()
        self.padding = nn.ReflectionPad2d(padding_size)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.softplus(x)
        x = self.padding(x)
        x = self.silu(x)
        return x


batch_size = 16
in_channels = 3
height, width = 1536, 1536  # Increased from 512 to 1536 to increase runtime
kernel_size = 3
padding_size = (1, 2, 1, 2)  # pad_left, pad_right, pad_top, pad_bottom


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, kernel_size, padding_size]