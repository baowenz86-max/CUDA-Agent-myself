import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, tensor_shape, dim):
        super().__init__()
        self.tensor = torch.nn.Parameter(torch.randn(*tensor_shape))
        self.dim = dim

    def forward(self, x):
        x = cuda_extension.cumprod(x, self.dim)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.cumprod(x, self.dim)
        yield 'step_001', x
        yield 'step_002', x
