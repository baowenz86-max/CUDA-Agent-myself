import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, start, end, step):
        super().__init__()
        self.sequence = torch.arange(start, end, step)

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        yield 'step_002', x
