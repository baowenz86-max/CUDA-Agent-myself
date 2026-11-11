import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = self.upsample(x)
        x = torch.mean(x, dim=1, keepdim=True)
        return x


batch_size = 128
channels = 3
height, width = 512, 256  # Increased width to increase computation
scale_factor = 2
mode = "bilinear"


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [scale_factor, mode]