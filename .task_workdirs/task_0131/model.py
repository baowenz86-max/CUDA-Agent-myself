import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.relu(x)
        x = nn.GELU()(x)
        return x


batch_size = 64
height, width = 2048, 2048  # Increased spatial dimensions to increase computation


def get_inputs():
    return [torch.randn(batch_size, 1, height, width)]


def get_init_inputs():
    # No arguments required for this model
    return []