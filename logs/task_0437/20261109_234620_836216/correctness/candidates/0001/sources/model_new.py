import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = cuda_extension.relu_add_div6(x)
        x = cuda_extension.hardswish(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.relu_add_div6(x)
        yield 'step_001', x
        x = cuda_extension.hardswish(x)
        yield 'step_002', x
        yield 'step_003', x
