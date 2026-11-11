import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape, test_elements_shape, complex_elements_shape):
        super().__init__()
        self.test_elements = nn.Parameter(torch.randn(test_elements_shape))
        self.complex_elements = nn.Parameter(
            torch.randn(complex_elements_shape) + 1j * torch.randn(complex_elements_shape)
        )

    def forward(self, x):
        x = cuda_extension.isin_forward(x, self.test_elements)
        y = cuda_extension.is_complex_forward(self.complex_elements)
        return x, y

    def debug_forward(self, x):
        z = cuda_extension.isin_forward(x, self.test_elements)
        yield 'step_001', z
        y = cuda_extension.is_complex_forward(self.complex_elements)
        yield 'step_002', y
        yield 'step_003', (z, y)
