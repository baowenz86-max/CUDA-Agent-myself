import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pad_size_h, pad_size_w):
        super().__init__()
        self.pad = nn.ReplicationPad2d((pad_size_h, pad_size_h, pad_size_w, pad_size_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pad(x)


pad_size_h = 2
pad_size_w = 3
batch_size = 16
channels = 3
height = 4096  # Increased from 512 to 4096 to increase computation
width = 2048   # Increased from 256 to 2048 to increase computation


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [pad_size_h, pad_size_w]