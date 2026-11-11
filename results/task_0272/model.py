import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, pooling_size):
        super().__init__()
        self.pooling_size = pooling_size

    def forward(self, x):
        x = torch.clamp(x, min=0.0, max=1.0)
        x = F.adaptive_avg_pool2d(x, self.pooling_size)
        return x


batch_size = 16
channels = 3
height = 4096  # Increased to increase computation
width = 2048   # Increased to increase computation
pooling_size = (16, 16)


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [pooling_size]