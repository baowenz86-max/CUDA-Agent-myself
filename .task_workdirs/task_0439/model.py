from collections import OrderedDict

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_modules, input_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.module_dict = nn.ModuleDict(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(input_channels, out_channels, kernel_size)),
                    ("relu", nn.ReLU()),
                    ("conv2", nn.Conv2d(out_channels, out_channels, kernel_size)),
                ]
            )
        )
        self.num_modules = num_modules

    def forward(self, x):
        x = self.module_dict["conv1"](x)
        for _ in range(self.num_modules):
            x = torch.abs(x)
            x = self.module_dict["relu"](x)
            x = self.module_dict["conv2"](x)
        return x


batch_size = 512
input_channels = 3
out_channels = 16
height, width = 64, 64  # Increased from 32x32 to 64x64 to increase computation
kernel_size = 3
num_modules = 2


def get_init_inputs():
    return [num_modules, input_channels, out_channels, kernel_size]


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]