import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.zero_pad = nn.ZeroPad2d(pad)

    def forward(self, x, elements):
        mask = torch.isin(x, elements)
        x = self.zero_pad(x)
        return x, mask


batch_size = 512
in_channels = 3
height, width = 128, 128  # Increased from 32x32 to 128x128
pad = (1, 1, 1, 1)
elements = torch.tensor([1.0, 2.0, 3.0])


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    elements = torch.tensor([1.0, 2.0, 3.0])
    return [x, elements]


def get_init_inputs():
    return [pad]