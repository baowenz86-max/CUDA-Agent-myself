import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad_width, pad_value):
        super().__init__()
        self.pad = nn.ConstantPad2d(pad_width, pad_value)

    def forward(self, x):
        x = self.pad(x)
        x = torch.max(x, dim=1)[0]
        return x


batch_size = 512  # Increased batch size to increase runtime
height, width = 512, 256
pad_width = (1, 2, 1, 2)
pad_value = 0


def get_inputs():
    x = torch.randn(batch_size, 1, height, width)
    return [x]


def get_init_inputs():
    return [pad_width, pad_value]