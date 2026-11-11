import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, channels, num_parameters):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters=num_parameters)

    def forward(self, x):
        x = torch.sum(x, dim=1, keepdim=True)  # Apply torch.sum to the 2nd dimension
        x = self.prelu(x)
        return x


def get_inputs():
    batch_size = 16
    channels = 3
    height = 2560
    width = 2560
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    channels = 3
    num_parameters = 1  # Since num_parameters is the number of DIVIDED channels, we use 1 for simplicity
    return [channels, num_parameters]