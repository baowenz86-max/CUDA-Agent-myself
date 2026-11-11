import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()

    def forward(self, x):
        x, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.view_as_complex(x.squeeze(1))
        return x


batch_size = 128
in_channels = 2
height, width = 512, 512  # Increased width to 512 to increase computation


def get_inputs():
    real_part = torch.randn(batch_size, in_channels, height, width)
    imag_part = torch.randn(batch_size, in_channels, height, width)
    x = torch.stack((real_part, imag_part), dim=-1)
    return [x]


def get_init_inputs():
    return [in_channels, height, width]