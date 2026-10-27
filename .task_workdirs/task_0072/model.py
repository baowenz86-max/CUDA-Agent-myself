import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = torch.mean(x, dim=self.input_size)
        x = F.gelu(x)
        return x


batch_size = 16
channels = 1024  # Increased from 3 to 1024 to increase computation
height = 512
width = 256


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [1]  # The dimension along which the mean operation will be applied