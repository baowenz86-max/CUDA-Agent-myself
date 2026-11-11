import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = torch.log2(x)
        return x


batch_size = 128
channels = 3
height, width = 1024, 256  # Increased spatial dimensions to increase runtime
padding = 2


def get_inputs():
    return [torch.rand(batch_size, channels, height, width) + 1e-5]  # Adding a small value to avoid log2(0)


def get_init_inputs():
    return [padding]