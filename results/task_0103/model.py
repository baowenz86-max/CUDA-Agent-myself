import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, repeats):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.repeats = repeats

    def forward(self, x):
        x = torch.tile(x, (self.repeats, 1, 1))  # 'tile' in PyTorch corresponds to 'repeat' in other frameworks
        x = self.conv1d(x)
        return x


batch_size = 64
in_channels = 128
out_channels = 64
kernel_size = 3
stride = 1
padding = 1
repeats = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, 4096)]  # Increased sequence length from 512 to 4096


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, repeats]