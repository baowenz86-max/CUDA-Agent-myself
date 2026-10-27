import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, padding_constant: int, padding_reflection: tuple):
        super().__init__()
        self.padding_constant = padding_constant
        self.padding_reflection = padding_reflection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.constant_pad2d(x, self.padding_constant)
        x = cuda_extension.reflection_pad2d(x, self.padding_reflection)
        return x

    def debug_forward(self, inputs):
        x = inputs[0]
        x = cuda_extension.constant_pad2d(x, self.padding_constant)
        yield 'step_001', x
        x = cuda_extension.reflection_pad2d(x, self.padding_reflection)
        yield 'step_002', x
        yield 'step_003', x
