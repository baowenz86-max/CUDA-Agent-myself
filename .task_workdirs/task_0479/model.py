import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pool_size, pooling_type):
        super().__init__()
        self.pooling_type = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}[pooling_type](kernel_size=pool_size)

    def forward(self, x):
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.pooling_type(x)
        return x


batch_size = 512
height, width = 320, 320  # Increased from 32x32 to 320x320
channels = 3
pool_size = (2, 2)
pooling_type = "max"  # or 'avg'


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [pool_size, pooling_type]