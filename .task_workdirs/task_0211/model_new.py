import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, div_factor):
        super().__init__()
        self.div_factor = div_factor

    def forward(self, x):
        x = cuda_extension.hardtanh_forward(x, -1.0, 1.0)
        x = cuda_extension.div_scalar_forward(x, self.div_factor)
        return x

    def debug_forward(self, x):
        x = cuda_extension.hardtanh_forward(x, -1.0, 1.0)
        yield 'step_001', x
        x = cuda_extension.div_scalar_forward(x, self.div_factor)
        yield 'step_002', x
        yield 'step_003', x
