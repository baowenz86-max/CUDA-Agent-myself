import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, eps=1e-5):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(input_size, eps=eps)

    def forward(self, x):
        x = self.instance_norm(x)
        x = torch.relu(x)
        return x


batch_size = 128
input_size = 3
height, width = 512, 512  # Increased width to 512 to increase computation


def get_inputs():
    return [torch.randn(batch_size, input_size, height, width)]


def get_init_inputs():
    return [input_size]