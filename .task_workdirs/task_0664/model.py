import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, complex_tensor, mean_axis):
        super().__init__()
        self.complex_tensor = complex_tensor
        self.mean_axis = mean_axis

    def forward(self, x):
        is_complex = torch.is_complex(self.complex_tensor)
        mean_tensor = torch.mean(x, dim=self.mean_axis)
        return mean_tensor, is_complex


batch_size, channels, height, width = 64, 3, 2048, 2048  # Increased spatial dimensions
complex_tensor = torch.tensor([1 + 1j, 1 + 1j], dtype=torch.cfloat)
mean_axis = 1


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [complex_tensor, mean_axis]