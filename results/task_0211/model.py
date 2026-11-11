import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, input_size, div_factor):
        super().__init__()
        self.div_factor = div_factor

    def forward(self, x):
        x = F.hardtanh(x)
        x = x / self.div_factor
        return x


batch_size = 512
in_channels = 3
height = 512
width = 512


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [14, 2.0]