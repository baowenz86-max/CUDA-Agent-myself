import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, num_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = F.hardswish(x)
        return x


batch_size = 512
height = 1024
width = 1024


def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]


def get_init_inputs():
    return [3, (3, 3), 1, 1]