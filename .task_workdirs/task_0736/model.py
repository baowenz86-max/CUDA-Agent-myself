import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dim):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.dim = dim

    def forward(self, x):
        x = F.softmax(x, dim=self.dim)
        x = F.conv1d(x, self.conv1d.weight, self.conv1d.bias, stride=self.conv1d.stride, padding=self.conv1d.padding)
        return x


batch_size = 512
in_channels = 3
out_channels = 32
length = 16384  # Increased from 512 to 16384 to increase computation
kernel_size = 3
stride = 1
padding = 1
dim = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dim]