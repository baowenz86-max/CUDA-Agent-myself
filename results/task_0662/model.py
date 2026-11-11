import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, mean_dim, padding_amount):
        super().__init__()
        self.padding = nn.ZeroPad2d(padding_amount)
        self.mean_dim = mean_dim

    def forward(self, x):
        x = self.padding(x)
        x = torch.mean(x, dim=self.mean_dim)
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
channels = 3
height = 512  # Increased from 256 to increase computation
width = 512   # Increased from 256 to increase computation
mean_dim = 2  # Can be either 1 (for across width) or 2 (for across height)
padding_amount = (1, 2, 1, 2)  # Padding for top, bottom, left, right respectively


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [mean_dim, padding_amount]