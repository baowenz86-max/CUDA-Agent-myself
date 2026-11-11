import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_channels, kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.input_channels = input_channels
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.log_softmax(x)
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x


batch_size = 512
input_channels = 5
height, width = 256, 256  # Increased from 64x64 to 256x256
kernel_size = 3
clamp_min = -1.0
clamp_max = 1.0


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [input_channels, kernel_size, clamp_min, clamp_max]