import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.softmax(x, dim=1)  # Softmax operation
        x = torch.relu(x)  # ReLU operation
        return x


batch_size = 16
width = 512
height = 2048  # Increased from 256 to increase computation


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return []