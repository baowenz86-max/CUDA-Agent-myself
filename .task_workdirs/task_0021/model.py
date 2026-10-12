import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv1d(x)
        x = torch.bmm(x, x.transpose(1, 2))
        x = F.softplus(x)
        x = x + self.bias
        return x


batch_size = 128
in_channels = 3
out_channels = 64
kernel_size = 3
bias_shape = (out_channels, 1)


def get_inputs():
    sequence_length = 8192  # Increased from 1024 to increase computation
    x = torch.randn(batch_size, in_channels, sequence_length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]