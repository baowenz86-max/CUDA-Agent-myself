import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad_size, input_channels, input_height, input_width):
        super().__init__()
        self.pad = nn.ReflectionPad2d(pad_size)
        self.input_height = input_height
        self.input_width = input_width

    def forward(self, x, y):
        x_padded = self.pad(x)
        y_padded = self.pad(y)
        max_val = torch.max(x_padded, y_padded)
        min_val = torch.minimum(max_val, x_padded)
        return min_val


batch_size = 128
input_channels = 3
input_height = 512
input_width = 512  # Increased from 64 to 512 to increase computational load
pad_size = 4


def get_inputs():
    x = torch.randn(batch_size, input_channels, input_height, input_width)
    y = torch.randn(batch_size, input_channels, input_height, input_width)
    return [x, y]


def get_init_inputs():
    return [pad_size, input_channels, input_height, input_width]