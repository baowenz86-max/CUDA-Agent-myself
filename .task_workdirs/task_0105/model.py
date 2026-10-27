import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x):
        x_real = torch.view_as_real(x)
        x_min = torch.minimum(x_real[:, 0], x_real[:, 1])
        return x_min


batch_size = 512
input_shape = (batch_size, 64, 128000)  # Increased sequence length to 128000
complex_shape = (batch_size, 64, 128000, 2)


def get_inputs():
    x = torch.randn(*complex_shape, dtype=torch.complex64)
    return [x]


def get_init_inputs():
    return [input_shape]