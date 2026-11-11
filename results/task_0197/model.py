import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.randn(512, 3, 320, 320)]  # Increased spatial dimensions to 320x320


def get_init_inputs():
    return []