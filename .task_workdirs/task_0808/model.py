import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, pad_value):
        super().__init__()
        self.pad_value = pad_value

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="constant", value=self.pad_value)
        x = torch.stack([x, x], dim=1)
        return x


batch_size, channels, height, width = 16, 3, 2048, 1024
pad_value = 0


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [pad_value]