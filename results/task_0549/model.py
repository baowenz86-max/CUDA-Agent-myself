import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, channel_num, negative_slope, max_value):
        super(Model, self).__init__()
        self.prelu = nn.PReLU(channel_num, init=negative_slope)
        self.max_value = max_value

    def forward(self, x):
        max_value_tensor = torch.full_like(x, self.max_value)
        x = torch.max(x, max_value_tensor)
        x = self.prelu(x)
        return x


batch_size = 512
channels = 64
height = 64
width = 64


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]  # Batch size: 512, Channels: 64, Height: 64, Width: 64


def get_init_inputs():
    return [64, 0.25, 2.0]  # Channels, negative_slope, max_value