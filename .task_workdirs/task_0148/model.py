import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, bins, padding_size):
        super().__init__()
        self.softmax = nn.Softmax(dim=input_size)
        self.histc = nn.Parameter(torch.zeros(bins))
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.softmax(x)
        x = torch.histc(x, bins=self.histc.size(0), min=0, max=1)
        return x


batch_size = 128
in_channels = 3
height, width = 512, 256  # Increased width to 256 to increase computation
input_size = 1  # Specify the dimension for Softmax
bins = 10  # Number of bins for torch.histc
padding_size = (1, 1, 1, 1)  # Padding size for nn.ReflectionPad2d


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [input_size, bins, padding_size]