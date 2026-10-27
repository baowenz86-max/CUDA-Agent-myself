import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = F.hardsigmoid(x)
        y = torch.relu(y)
        return x, y


batch_size = 128
features = 64
input_width, input_height = 1024, 32  # Increased input_width to 1024


def get_inputs():
    x = torch.randn(batch_size, features, input_height, input_width)
    y = torch.randn(batch_size, features, input_height, input_width)
    return [x, y]


def get_init_inputs():
    return []