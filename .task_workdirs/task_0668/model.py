import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes, divisor):
        super().__init__()
        self.divisor = divisor

    def forward(self, x):
        x = torch.log_softmax(x, dim=1)
        x = x / self.divisor
        return x


batch_size = 64
num_classes = 10
height = 1024  # Increased from 512 to 1024 to increase computation
width = 256
divisor = 0.5


def get_inputs():
    x = torch.randn(batch_size, num_classes, height, width)
    return [x]


def get_init_inputs():
    return [num_classes, divisor]