import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x):
        x = cuda_extension.tril(x)
        return cuda_extension.relu(x)

    def debug_forward(self, x):
        x = cuda_extension.tril(x)
        yield 'step_001', x
        x = cuda_extension.relu(x)
        yield 'step_002', x
