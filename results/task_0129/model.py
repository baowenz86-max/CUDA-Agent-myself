import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.softmax(x, dim=1)
        x = torch.cumprod(x, dim=2)
        return x


batch_size = 512
channels = 16
seq_len = 16384  # Increased from 64 to 16384 to increase computation


def get_inputs():
    # Shape of the input tensor is [batch_size, channels, seq_len]
    x = torch.randn(batch_size, channels, seq_len)
    return [x]


def get_init_inputs():
    # No parameters needed for initialization
    return []