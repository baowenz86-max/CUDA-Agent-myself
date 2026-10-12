import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, dim, indices):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size)
        self.relu = nn.ReLU()
        self.index_select = indices

    def forward(self, x):
        x = self.pad(x)
        x = self.relu(x)
        x = torch.index_select(x, self.index_select, torch.arange(x.size(self.index_select)).to(x.device))
        return x


batch_size = 512
in_channels = 3
height, width = 170, 170  # Increased from 32x32 to 170x170
kernel_size = 3
dim = 2
indices = 0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, kernel_size, dim, indices]