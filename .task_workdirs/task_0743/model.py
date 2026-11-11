import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.unfold = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = self.unfold(x)
        return x


batch_size = 512
channels = 3
height = 128  # Increased from 64 to 128 to increase runtime
width = 128   # Increased from 64 to 128 to increase runtime
upscale_factor = 2


def get_inputs():
    x = torch.randn(batch_size, channels * upscale_factor**2, height, width)
    return [x]


def get_init_inputs():
    return [upscale_factor]