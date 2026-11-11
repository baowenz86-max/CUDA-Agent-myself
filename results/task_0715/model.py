import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_parameters, use_relu):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters, use_relu)

    def forward(self, x):
        x = torch.abs(x)  # torch.abs
        x = self.prelu(x)  # nn.PReLU
        return x


batch_size = 16
channels = 64  # Increased from 3 to 64 to increase computation
width = 512
height = 256
num_parameters = channels
use_relu = False


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [num_parameters, use_relu]