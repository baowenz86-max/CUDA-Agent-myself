import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = self.pool(x)
        return x


batch_size = 128
in_channels = 32
sequence_length = 15360  # Increased to increase computation time
kernel_size = 3
stride = 2
padding = 1
dilation = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, sequence_length)]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]