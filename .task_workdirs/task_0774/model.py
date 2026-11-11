import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size, einsum_equation):
        super().__init__()
        self.einsum_equation = einsum_equation

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = torch.einsum(self.einsum_equation, x)
        return x


batch_size = 512
height = 128
width = 128
channels = 64  # Increased from 3 to 64 to increase computation
input_size = (batch_size, channels, height, width)
einsum_equation = "bchw->bhwc"


def get_inputs():
    return [torch.randn(input_size)]


def get_init_inputs():
    return [input_size[1], einsum_equation]