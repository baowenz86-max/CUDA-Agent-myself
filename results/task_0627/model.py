import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad_size, channels):
        super().__init__()
        self.pad2d = nn.ZeroPad2d(pad_size)
        self.channels = channels

    def forward(self, x):
        x = self.pad2d(x)
        x = torch.sum(x, dim=1, keepdim=True)  # sum across channels
        return x


# Set your data here
batch_size = 256  # Increased from 16 to 256 to increase runtime
in_channels = 3
out_channels = 64
height = 512
width = 256
pad_size = (1, 2, 3, 4)  # pad_size = (pad_left, pad_right, pad_top, pad_bottom)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [pad_size, in_channels]