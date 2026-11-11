import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, num_channels, stack_dim):
        super().__init__()
        self.stack_dim = stack_dim
        self.relu = torch.relu
        self.input_size = input_size
        self.num_channels = num_channels
        self.stack = nn.Parameter(
            torch.stack([torch.randn(self.input_size) for _ in range(self.num_channels)], dim=self.stack_dim)
        )

    def forward(self, x):
        out = cuda_extension.stack_relu_forward(x, self.num_channels, self.stack_dim)
        return out

    def debug_forward(self, x):
        out = cuda_extension.stack_relu_forward(x, self.num_channels, self.stack_dim)
        yield 'step_001', out
        yield 'step_002', out
