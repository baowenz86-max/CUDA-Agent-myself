import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)

    def forward(self, x, y):
        x = self.pad(x)
        x = torch.bitwise_xor(x, y)
        return x


batch_size = 128
channels = 3
height, width = 512, 512  # Increased width from 32 to 512 to increase computational load
padding = 2


def get_inputs():
    x = torch.randint(0, 256, (batch_size, channels, height, width), dtype=torch.uint8)
    y = torch.randint(0, 256, (batch_size, channels, height + 2 * padding, width + 2 * padding), dtype=torch.uint8)
    return [x, y]


def get_init_inputs():
    return [padding]