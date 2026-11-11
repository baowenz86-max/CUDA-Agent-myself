import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        x = cuda_extension.cos_forward(x)
        x = self.identity(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.cos_forward(x)
        yield 'step_001', x
        x = self.identity(x)
        yield 'step_002', x
        yield 'step_003', x
