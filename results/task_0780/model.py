import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, data_shape, values_shape):
        super().__init__()
        self.data = nn.ParameterList([nn.Parameter(torch.randn(data_shape)) for _ in range(2)])
        self.values = nn.Parameter(torch.randn(values_shape))

    def forward(self, x):
        x = torch.isin(x, self.values)
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
data_shape = (128, 128)  # Increased from (32,32) to increase computation
values_shape = (1000,)  # Increased from (100,) to increase computation


def get_inputs():
    return [torch.randn(batch_size, *data_shape)]


def get_init_inputs():
    return [data_shape, values_shape]