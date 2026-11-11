import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, input_size, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        x = cuda_extension.softmax_dim1(x, self.temperature)
        x = cuda_extension.mul_square(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.softmax_dim1(x, self.temperature)
        yield "step_001", x
        x = cuda_extension.mul_square(x)
        yield "step_002", x
        yield "step_003", x
