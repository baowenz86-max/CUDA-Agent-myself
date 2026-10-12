import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, padding_size):
        super().__init__()
        # Store padding_size as tuple; kernel_size is unused in this reference model
        self.padding_size = tuple(padding_size)

    def forward(self, x):
        import cuda_extension
        x = cuda_extension.reflection_pad2d_forward(x, self.padding_size)
        x = cuda_extension.ge_scalar_forward(x, 0.0)
        x = cuda_extension.remainder_scalar_forward(x, 2.0)
        return x

    def debug_forward(self, *inputs):
        import cuda_extension
        x = inputs[0]
        padded = cuda_extension.reflection_pad2d_forward(x, self.padding_size)
        yield 'step_001', padded

        ge_out = cuda_extension.ge_scalar_forward(padded, 0.0)
        yield 'step_002', ge_out

        remainder_out = cuda_extension.remainder_scalar_forward(ge_out, 2.0)
        yield 'step_003', remainder_out

        yield 'step_004', remainder_out
