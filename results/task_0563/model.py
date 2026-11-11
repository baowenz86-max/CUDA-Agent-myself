import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, tensor_a, tensor_b):
        super().__init__()
        self.relu = nn.ReLU()
        self.tensor_a = nn.Parameter(tensor_a)
        self.tensor_b = nn.Parameter(tensor_b)

    def forward(self, x):
        x = self.relu(x)
        equal_result = torch.equal(self.tensor_a, self.tensor_b)
        return x, equal_result


batch_size = 16
channels = 3
height = 4096
width = 4096


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    tensor_a = torch.randn(64)
    tensor_b = torch.randn(64)
    return [tensor_a, tensor_b]