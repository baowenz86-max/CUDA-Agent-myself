import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.nan_to_num = torch.nan_to_num

    def forward(self, x):
        x = torch.sum(x, dim=1, keepdim=True)
        x = self.nan_to_num(x)
        return x


batch_size = 16
channels = 3
height = 4096  # Increased to 4096 to increase computation
width = 4096   # Increased to 4096 to increase computation


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    # Introduce some nans for testing purposes
    x[0, 0, 0, 0] = float("nan")
    return [x]


def get_init_inputs():
    return []