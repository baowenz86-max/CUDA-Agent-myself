import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = torch.detach(x)  # Detach the tensor
        x = torch.unbind(x, dim=-1)  # Unbind along last dimension
        x = [torch.max(xi, dim=1) for xi in x]  # Compute max along second axis
        return x


# Hyperparameters determining test input shape
batch_size = 512  # Increased to ensure at least one dimension >=512
height = 128
width = 512  # Increased width to increase computation


def get_inputs():
    # Generate random tensor (batch_size, input_size, height, width)
    x = torch.randn(batch_size, *[3, height, width])
    return [x]


def get_init_inputs():
    return [3]  # Assuming input_size is hardcoded to 3