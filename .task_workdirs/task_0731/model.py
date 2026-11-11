import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        x = F.softmax(x / self.temperature, dim=1)
        x = torch.mul(x, x)
        return x


batch_size = 16
height, width = 32, 32
input_size = 1024  # Increased from 512 to 1024 to increase computational load
temperature = 0.5


def get_inputs():
    return [torch.randn(batch_size, input_size, height, width)]


def get_init_inputs():
    return [input_size, temperature]