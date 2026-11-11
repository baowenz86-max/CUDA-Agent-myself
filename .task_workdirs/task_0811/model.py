import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale_factor, mode, softmax_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.softmax_dim = softmax_dim

    def forward(self, x):
        x = self.upsample(x)
        x = torch.softmax(x, dim=self.softmax_dim)
        return x


batch_size = 64
channels = 3
height, width = 512, 512  # Increased width to 512 to increase computational load
scale_factor = 2.0
mode = "bilinear"
softmax_dim = 1


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [scale_factor, mode, softmax_dim]