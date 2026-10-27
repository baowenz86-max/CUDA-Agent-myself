import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, padding):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.padding = nn.ReflectionPad2d(padding)

    def forward(self, x):
        x = torch.log_softmax(x, dim=1)
        x = self.softmax(x)
        x = self.padding(x)
        return x


batch_size = 512
in_channels = 3
height, width = 256, 256  # Increased from 64x64 to 256x256
padding = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, padding]