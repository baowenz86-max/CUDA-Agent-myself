import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, x):
        # Step 001: x = torch.abs(x)
        x = cuda_extension.abs_forward(x)
        # Step 002: x = torch.unsqueeze(x, dim=self.expand_dim)
        x = cuda_extension.unsqueeze_forward(x, self.expand_dim)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        # Replicate forward steps exactly
        # Step 001: x = torch.abs(x)
        x_abs = cuda_extension.abs_forward(x)
        yield 'step_001', x_abs
        # Step 002: x = torch.unsqueeze(x, dim=self.expand_dim)
        x_unsqueezed = cuda_extension.unsqueeze_forward(x_abs, self.expand_dim)
        yield 'step_002', x_unsqueezed
        # Step 003: return x
        yield 'step_003', x_unsqueezed


batch_size = 512
width = 1024
height = 1024
input_size = (batch_size, height, width)
expand_dim = 1


def get_inputs():
    x = torch.randn(*input_size)
    return [x]


def get_init_inputs():
    return [input_size, expand_dim]
