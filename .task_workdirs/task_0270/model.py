import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, pad_width, pad_mode):
        super().__init__()
        self.pad_width = pad_width
        self.pad_mode = pad_mode

    def forward(self, x):
        x = torch.deg2rad(x)
        x = F.pad(x, self.pad_width, mode=self.pad_mode)
        return x


batch_size = 128
height, width = 4096, 256  # Increased from (512, 32) to increase computation
pad_width = (2, 2, 1, 1)
pad_mode = "reflect"


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return [pad_width, pad_mode]