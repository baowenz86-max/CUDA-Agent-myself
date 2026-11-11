import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=input_size, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.sum(x, dim=1, keepdim=True)
        return x


batch_size = 512
in_channels = 1
height = 224
width = 224


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [1, 4, (3, 3), 1, 1]