import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_channels, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        y = cuda_extension.max_pool2d_forward(x, self.pool_size, self.pool_size)
        return cuda_extension.stack_forward([y])

    def debug_forward(self, x):
        y = cuda_extension.max_pool2d_forward(x, self.pool_size, self.pool_size)
        stack = [y]
        yield 'step_001', stack
        z = cuda_extension.stack_forward(stack)
        yield 'step_002', z
