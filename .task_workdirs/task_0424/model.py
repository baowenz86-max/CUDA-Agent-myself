import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_parameters, num_channels, kernel_size, bias=True):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters=num_parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(x)


batch_size = 512
num_channels = 3
height = 512
width = 512


def get_inputs():
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    # Assuming num_parameters is the number of channels and bias is always True
    return [3, 3, (1, 1), True]