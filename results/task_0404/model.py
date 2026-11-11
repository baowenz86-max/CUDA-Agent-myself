import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, binary_threshold):
        super().__init__()
        self.binary_threshold = binary_threshold

    def forward(self, x):
        x = torch.abs(x)
        x = torch.where(x > self.binary_threshold, x, torch.tensor(0.0))
        return x


batch_size = 512
height, width = 512, 512  # Increased from 64x64 to 512x512
input_size = (batch_size, 1, height, width)
binary_threshold = 0.5


def get_inputs():
    return [torch.randn(input_size)]


def get_init_inputs():
    return [input_size[1], binary_threshold]