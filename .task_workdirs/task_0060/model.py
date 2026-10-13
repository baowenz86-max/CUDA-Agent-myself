import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # The forward method will include ReLU and HardSigmoid in the order received from the user input

    def forward(self, x):
        x = torch.relu(x)
        x = F.hardsigmoid(x)
        return x


def get_inputs():
    # Random input tensor with increased size to increase runtime
    return [torch.randn(256, 1024, 1024)]


def get_init_inputs():
    # No parameters needed for this case
    return []