import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad_width, values):
        super().__init__()
        self.pad = nn.ConstantPad1d(pad_width, values)

    def forward(self, x):
        x = self.pad(x)
        x = torch.sum(x, dim=2)
        return x


batch_size = 128
seq_length = 512
width = 10000  # Increased from 10 to 10000 to increase runtime
pad_width = (2, 2)
values = 0.0


def get_inputs():
    return [torch.randn(batch_size, seq_length, width)]


def get_init_inputs():
    return [pad_width, values]