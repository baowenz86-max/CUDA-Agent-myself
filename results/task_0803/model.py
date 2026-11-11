import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape, test_elements_shape, complex_elements_shape):
        super().__init__()
        self.test_elements = nn.Parameter(torch.randn(test_elements_shape))
        self.complex_elements = nn.Parameter(
            torch.randn(complex_elements_shape) + 1j * torch.randn(complex_elements_shape)
        )

    def forward(self, x):
        x = torch.isin(x, self.test_elements)
        y = torch.is_complex(self.complex_elements)
        return x, y


batch_size = 2048
input_shape = (batch_size, 2048)
test_elements_shape = (1024,)
complex_elements_shape = (batch_size, 128)


def get_inputs():
    return [torch.randn(input_shape)]


def get_init_inputs():
    return [input_shape, test_elements_shape, complex_elements_shape]