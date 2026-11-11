import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, mask_value):
        super().__init__()
        self.mask_value = mask_value

    def forward(self, x):
        # torch.masked_fill: Fills elements of self tensor with value where mask is True.
        # torch.cat: Concatenates the given sequence of seq tensors along the given dimension dim.
        x = torch.masked_fill(x, x < self.mask_value, 0)
        x = torch.cat([x, x], dim=1)  # Assuming we want to double the output channels
        return x


batch_size = 16
in_channels = 32  # Increased from 3 to 32 to increase computation
out_channels = 2
height = 512
width = 256
mask_value = 0.5


def get_inputs():
    # Create a random tensor with values between 0 and 1, to represent a feature map.
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, mask_value]