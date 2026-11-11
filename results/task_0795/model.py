import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, num_elements):
        super().__init__()
        self.input_size = input_size
        self.num_elements = num_elements

    def forward(self, x):
        x = torch.stack([x] * self.num_elements, dim=self.input_size)
        x = torch.relu(x)
        return x


batch_size = 16
input_size = 1
width = 256
height = 512  # Adjusted height to 512 to ensure one dimension >=512
num_elements = 100  # Increased num_elements to increase computation


def get_inputs():
    x = torch.randn(batch_size, input_size, height, width)
    return [x]


def get_init_inputs():
    return [input_size, num_elements]