import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log_softmax(x, dim=1)
        x = torch.min(x, dim=1)[0]
        return x


batch_size = 64
width = 128
height = 2048  # Increased from 512 to increase computation


def get_inputs():
    # Generate random input data with the specified shape
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    # No initialization parameters needed for this model
    return []