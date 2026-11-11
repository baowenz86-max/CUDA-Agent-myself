import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, norm_type, kernel_size, stride):
        super().__init__()
        self.lppool = nn.LPPool1d(norm_type, kernel_size, stride=stride)

    def forward(self, x):
        return self.lppool(x)


batch_size = 128
channels = 32
length = 16384  # Increased from 512 to 16384 to increase runtime
norm_type = 2
kernel_size = 3
stride = 2


def get_inputs():
    x = torch.randn(batch_size, channels, length)
    return [x]


def get_init_inputs():
    return [norm_type, kernel_size, stride]